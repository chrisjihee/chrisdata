import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments, TableOption, IndexOption, ElasticStreamer
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class DataOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    total: int = field(default=-1)
    start: int = field(default=0)
    limit: int = field(default=-1)
    batch: int = field(default=1)
    logging: int = field(default=10000)
    from_table: bool = field(default=False)

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)


@dataclass
class ProgramArguments(CommonArguments):
    data: DataOption = field()
    table: TableOption = field()
    index: IndexOption = field()

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
            to_dataframe(columns=columns, raw=self.table, data_prefix="table"),
            to_dataframe(columns=columns, raw=self.index, data_prefix="index"),
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


def print_search_results(response: ObjectApiResponse):
    if response.meta.status:
        res = response.body
        logger.info(f"Got {res['hits']['total']['value']} hits (Max={res['hits']['max_score']}):")
        for hit in res["hits"]["hits"]:
            logger.info(f"  - {hit}")
        return res['hits']['total']['value']
    return 0


@app.command()
def search(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="search_wikidata"),
        output_home: str = typer.Option(default="output-search_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # data
        data_home: str = typer.Option(default="input/Wikidata-parse"),
        data_name: str = typer.Option(default="wikipedia-20230920-parse-kowiki.jsonl.bz2"),
        data_total: int = typer.Option(default=2009624),
        data_start: int = typer.Option(default=0),
        data_limit: int = typer.Option(default=-1),
        data_batch: int = typer.Option(default=10000),
        data_logging: int = typer.Option(default=100000),
        from_table: bool = typer.Option(default=True),
        # table
        table_host: str = typer.Option(default="localhost:6382"),
        # index
        index_host: str = typer.Option(default="localhost:9810"),
        index_user: str = typer.Option(default="elastic"),
        index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
        index_reset: bool = typer.Option(default=True),
        index_create: str = typer.Option(default="input/Wikidata-parse/wikipedia-index_create_opt.json"),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        logging_home=output_home,
        logging_file=logging_file,
        message_level=logging.DEBUG if debugging else logging.INFO,
        message_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_36,
    )
    data_opt = DataOption(
        home=data_home,
        name=data_name,
        total=data_total,
        start=data_start,
        limit=data_limit,
        batch=data_batch,
        logging=data_logging,
        from_table=from_table,
    )
    table_name = data_opt.name.stem.removesuffix(".jsonl")
    table_opt = TableOption(
        home=table_host,
        sect=env.project,
        name=table_name,
    )
    index_name = data_opt.name.stem.replace("-parse-", "-index-").replace(".jsonl", "")
    index_opt = IndexOption(
        home=index_host,
        user=index_user,
        pswd=index_pswd,
        name=index_name,
        reset=index_reset,
        create=index_create,
    )
    args = ProgramArguments(
        env=env,
        data=data_opt,
        table=table_opt,
        index=index_opt,
    )
    tqdm = mute_tqdm_cls()

    logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
    with (JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='=')):
        with ElasticStreamer(args.index) as client:
            # Search a query
            query1, query2, nbest = '63빌딩', '마천루', 5
            hc = print_search_results(client.search(
                index=args.index.name,
                query={
                    "query_string": {
                        "default_field": "body_text",
                        "query": f'"{query1}" AND "{query2}"'
                    },
                },
                _source=("_id", "title", "subtitle1", "subtitle2", "body_text"),
                size=nbest,
            ))
            h1 = print_search_results(client.search(
                index=args.index.name,
                query={
                    "query_string": {
                        "default_field": "body_text",
                        "query": f'"{query1}"'
                    },
                },
                _source=("_id", "title", "subtitle1", "subtitle2", "body_text"),
                size=nbest,
            ))
            h2 = print_search_results(client.search(
                index=args.index.name,
                query={
                    "query_string": {
                        "default_field": "body_text",
                        "query": f'"{query2}"'
                    },
                },
                _source=("_id", "title", "subtitle1", "subtitle2", "body_text"),
                size=nbest,
            ))
            N = 2009624
            n = hc / N
            d = (h1 / N) * (h2 / N)
            e = 0.0000001
            import math
            pmi = math.log2((n + e) / (d + e))
            logger.critical(f"PMI({query1}, {query2}): {pmi}")

            print_search_results(client.search(
                index=args.index.name,
                query={
                    "match": {
                        "_id": "0051525-001-002-001",
                    },
                },
                _source=("_id", "title", "subtitle1", "subtitle2", "body_text"),
                size=nbest,
            ))

            print_search_results(client.search(
                index=args.index.name,
                query={
                    "query_string": {
                        "default_field": "title",
                        "query": "서울*",
                    },
                },
                _source=("_id", "title", "subtitle1", "subtitle2", "body_text"),
                size=nbest,
            ))

            print_search_results(client.search(
                index=args.index.name,
                query={
                    "match": {
                        "title": "서울특별시",
                    },
                },
                _source=("_id", "title", "subtitle1", "subtitle2", "body_text"),
                size=nbest,
            ))


if __name__ == "__main__":
    app()
