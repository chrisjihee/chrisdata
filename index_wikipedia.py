import logging
import math
from dataclasses import dataclass, field
from itertools import islice
from typing import Iterable

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from more_itertools import ichunked

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, CommonArguments
from chrisbase.data import DataOption, FileOption, TableOption, IndexOption, MongoDBWrapper, ElasticSearchWrapper
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class ProgramArguments(CommonArguments):
    data: DataOption = field()

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
        ]).reset_index(drop=True)


@dataclass
class PassageUnit(DataClassJsonMixin):
    _id: str
    title: str
    subtitle1: str
    subtitle2: str
    body_text: str


def process_many(batch: Iterable[dict], batch_size: int, index_name: str, client: Elasticsearch):
    for ok, action in streaming_bulk(client=client,
                                     index=index_name,
                                     actions=batch,
                                     chunk_size=batch_size,
                                     yield_ok=False):
        logger.warning(f"ok={ok}, action={action}")


@app.command()
def index(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="index_wikipedia"),
        output_home: str = typer.Option(default="output-index_wikipedia"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # data
        data_start: int = typer.Option(default=0),
        data_limit: int = typer.Option(default=-1),
        data_batch: int = typer.Option(default=10000),
        data_inter: int = typer.Option(default=100000),
        data_total: int = typer.Option(default=2009624),
        file_home: str = typer.Option(default="input/wikimedia"),
        file_name: str = typer.Option(default="wikipedia-20230920-parse-kowiki.jsonl.bz2"),
        table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        table_name: str = typer.Option(default="wikipedia-20230920-parse-kowiki"),
        index_home: str = typer.Option(default="localhost:9810"),
        index_name: str = typer.Option(default="wikipedia-20230920-index-kowiki"),
        index_user: str = typer.Option(default="elastic"),
        index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
        index_reset: bool = typer.Option(default=True),
        index_create: str = typer.Option(default="input/wikimedia/wikipedia-index_create_args.json"),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        output_home=output_home,
        logging_file=logging_file,
        msg_level=logging.DEBUG if debugging else logging.INFO,
        msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_36,
    )
    data_opt = DataOption(
        start=data_start,
        limit=data_limit,
        batch=data_batch,
        inter=data_inter,
        total=data_total,
        file=FileOption(
            home=file_home,
            name=file_name,
        ) if file_home and file_name else None,
        table=TableOption(
            home=table_home,
            name=table_name,
        ) if table_home and table_name else None,
        index=IndexOption(
            home=index_home,
            user=index_user,
            pswd=index_pswd,
            name=index_name,
            reset=index_reset,
            create=index_create,
        ) if index_home and index_name else None,
    )
    args = ProgramArguments(
        env=env,
        data=data_opt,
    )
    tqdm = mute_tqdm_cls()

    logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
    with (
        JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        ElasticSearchWrapper(args.data.index) as data_index,
        MongoDBWrapper(args.data.table) as data_table,
    ):
        # Prepare an index
        if args.data.index.reset:
            if data_index.indices.exists(index=args.data.index.name):
                data_index.indices.delete(index=args.data.index.name)
            data_index.indices.create(index=args.data.index.name, **args.data.index.create_args)
            logger.info(f"Created a new index: {args.data.index}")
            logger.info(f"- Option: keys={list(args.data.index.create_args.keys())}")

        exit(1)

        # Load and Indexing documents
        if args.data.from_table:
            logger.info(f"Use database table: {args.table}")
            find_opt = {}
            num_input, inputs = data_table.count_documents(find_opt), data_table.find(find_opt).sort("_id")
        else:
            num_input, inputs = args.data.total, iter_compressed(args.data.home / args.data.name)
            inputs = map(PassageUnit.to_dict, map(PassageUnit.from_json, inputs))
        if args.data.start > 0:
            num_input, inputs = max(0, min(num_input, num_input - args.data.start)), islice(inputs, args.data.start, num_input)
        if args.data.limit > 0:
            num_input, inputs = min(num_input, args.data.limit), islice(inputs, args.data.limit)
        num_batch, batches = math.ceil(num_input / args.data.batch), ichunked(inputs, args.data.batch)
        logger.info(f"Index {num_input} inputs with {num_batch} batches to {args.data.index}")
        progress, interval = (tqdm(batches, total=num_batch, unit="batch", pre="*", desc="indexing"),
                              math.ceil(args.data.inter / args.data.batch))
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            process_many(batch=x, batch_size=args.data.batch, index_name=args.data.index.name, client=data_index)
        logger.info(progress)
        data_index.indices.refresh(index=args.data.index.name)
        for line in str(data_index.cat.indices(index=args.data.index.name, v=True).body.strip()).splitlines():
            logger.info(line)

        # Search a query
        query1, query2, nbest = '지미 카터', '대한민국', 100
        response: ObjectApiResponse = data_index.search(
            index=args.data.index.name,
            query={
                "query_string": {
                    "default_field": "body_text",
                    "query": f'"{query1}" AND "{query2}"'
                },
            },
            _source=("_id", "title", "subtitle1", "subtitle2", "body_text"),
            size=nbest,
        )
        if response.meta.status:
            res = response.body
            logger.info(f"Got {res['hits']['total']['value']} hits (Max={res['hits']['max_score']}):")
            for hit in res["hits"]["hits"]:
                logger.info(f"  - {hit}")


if __name__ == "__main__":
    app()
