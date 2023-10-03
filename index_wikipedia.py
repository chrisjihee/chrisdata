import logging
import math
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from more_itertools import ichunked

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments, TableOption, IndexOption, MongoDBTable, ElasticSearchClient
from chrisbase.io import LoggingFormat, iter_compressed
from chrisbase.util import to_dataframe, mute_tqdm_cls

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class DataOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    total: int = field(default=9740173)  # wc -l Wikipedia-20230920-parse-kowiki.jsonl
    start: int = field(default=0)
    limit: int = field(default=-1)
    batch: int = field(default=1)
    from_table: bool = field(default=False)
    prog_interval: int = field(default=10000)

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)


@dataclass
class ProgramArguments(CommonArguments):
    data: DataOption = field()
    table: TableOption = field()
    index: IndexOption = field()
    other: str | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
            to_dataframe(columns=columns, raw=self.index, data_prefix="index"),
            to_dataframe(columns=columns, raw={"other": self.other}),
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
                                     # actions=_batch_iter(batch=x),
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
        input_home: str = typer.Option(default="input/Wikidata-parse"),
        input_name: str = typer.Option(default="Wikipedia-20230920-parse-kowiki.jsonl.bz2"),
        input_total: int = typer.Option(default=9740173),
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=10000),
        prog_interval: int = typer.Option(default=100000),
        from_table: bool = typer.Option(default=False),
        # table
        db_host: str = typer.Option(default="localhost:6382"),
        # index
        index_host: str = typer.Option(default="localhost:9717"),
        index_user: str = typer.Option(default="elastic"),
        index_pswd: str = typer.Option(default="HOExBs8qAzdL3gUEdEq2"),
        index_create_opt: str = typer.Option(default="input/Wikidata-parse/Wikipedia-20230920-parse-kowiki-index_create_opt.json"),
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
        home=input_home,
        name=input_name,
        total=input_total,
        start=input_start,
        limit=input_limit,
        batch=input_batch,
        from_table=from_table,
        prog_interval=prog_interval,
    )
    table_name = data_opt.name.stem.replace(".jsonl", "")
    table_opt = TableOption(
        db_host=db_host,
        db_name=env.project,
        tab_name=table_name,
    )
    index_name = data_opt.name.stem.replace("-parse-", "-index-").replace(".jsonl", "").lower()
    index_opt = IndexOption(
        host=index_host,
        user=index_user,
        pswd=index_pswd,
        name=index_name,
        create_opt=index_create_opt,
    )
    args = ProgramArguments(
        env=env,
        data=data_opt,
        table=table_opt,
        index=index_opt,
    )
    tqdm = mute_tqdm_cls()

    logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
    with (JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='=')):
        with MongoDBTable(args.table) as inp_table, ElasticSearchClient(args.index) as out_client:
            if args.data.from_table:
                logger.info(f"Use database table: {args.table}")
                find_opt = {}
                num_input, inputs = inp_table.count_documents(find_opt), inp_table.find(find_opt).sort("_id")
            else:
                num_input, inputs = args.data.total, iter_compressed(args.data.home / args.data.name)
                inputs = map(PassageUnit.to_dict, map(PassageUnit.from_json, inputs))
            if args.data.start > 0:
                num_input, inputs = max(0, min(num_input, num_input - args.data.start)), islice(inputs, args.data.start, num_input)
            if args.data.limit > 0:
                num_input, inputs = min(num_input, args.data.limit), islice(inputs, args.data.limit)
            num_batch, batches = math.ceil(num_input / args.data.batch), ichunked(inputs, args.data.batch)
            logger.info(f"Index {num_input} inputs with {num_batch} batches to {args.index}")
            progress, interval = (tqdm(batches, total=num_batch, unit="batch", pre="*", desc="indexing"),
                                  math.ceil(args.data.prog_interval / args.data.batch))

            if out_client.indices.exists(index=args.index.name):
                logger.info(f"Delete an existing index: {args.index}")
                out_client.indices.delete(index=args.index.name)
            out_client.indices.create(index=args.index.name, **args.index.create_opt)
            logger.info(f"Created a new index: {args.index}")

            for i, x in enumerate(progress):
                if i > 0 and i % interval == 0:
                    logger.info(progress)
                process_many(batch=x, batch_size=args.data.batch, index_name=args.index.name, client=out_client)
            logger.info(progress)
            out_client.indices.refresh(index=args.index.name)
            for line in str(out_client.cat.indices(index=args.index.name, v=True).body.strip()).splitlines():
                logger.info(line)

            query, nbest = '카터 * 대한민국', 10
            response: ObjectApiResponse = out_client.search(
                index=args.index.name,
                query={
                    "match": {
                        "body_text": {"query": query}
                    },
                },
                _source=("_id", "title", "subtitle1", "subtitle2", "body_text"),
                size=nbest,
            )
            if response.meta.status:
                res = response.body
                logger.info("Got %d Hits (Max=%.3f):", res['hits']['total']['value'], res['hits']['max_score'])
                for hit in res["hits"]["hits"]:
                    logger.info(f"  - {hit}")


if __name__ == "__main__":
    app()
logging-1003.180159.out