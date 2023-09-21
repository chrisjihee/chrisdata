import json
import logging
import math
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from more_itertools import ichunked
from pymongo.collection import Collection
from qwikidata.claim import WikidataClaim
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme, ClaimsMixin
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.typedefs import LanguageCode

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments, TableOption, MongoDBTable
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class DataOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    total: int = field(default=106781030)  # https://www.wikidata.org/wiki/Wikidata:Statistics
    limit: int = field(default=-1)
    batch: int = field(default=1)
    from_scratch: bool = field(default=False)
    prog_interval: int = field(default=10000)

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)


@dataclass
class IndexOption(OptionData):
    host: str = field()
    cert: str = field()
    pswd: str = field()
    name: str = field()

    def __repr__(self):
        return f"{self.host}/{self.name}"


@dataclass
class ProgramArguments(CommonArguments):
    data: DataOption = field()
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
        input_name: str = typer.Option(default="Wikipedia-20230920-crawl-kowiki.jsonl.bz2"),
        input_total: int = typer.Option(default=1410203),
        input_limit: int = typer.Option(default=10),
        input_batch: int = typer.Option(default=10),
        from_scratch: bool = typer.Option(default=True),
        prog_interval: int = typer.Option(default=10),
        # index
        index_host: str = typer.Option(default="localhost:9200"),
        index_cert: str = typer.Option(default="cfg/http_ca.crt"),
        index_pass: str = typer.Option(default="cfg/eleastic-pw.txt"),
        index_name: str = typer.Option(default="Wikipedia-20230920-crawl-kowiki"),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        output_home=output_home,
        logging_file=logging_file,
        msg_level=logging.DEBUG if debugging else logging.INFO,
        msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
    )
    args = ProgramArguments(
        env=env,
        data=DataOption(
            home=input_home,
            name=input_name,
            total=input_total,
            limit=input_limit,
            batch=input_batch,
            from_scratch=from_scratch,
            prog_interval=prog_interval,
        ),
        index=IndexOption(
            host=index_host,
            cert=index_cert,
            pswd=index_pass,
            name=index_name,
        ),
    )
    tqdm = mute_tqdm_cls()
    output_file = (args.env.output_home / f"{args.data.name.stem}-{args.env.time_stamp}.jsonl")

    with (JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='=')):
        pass


if __name__ == "__main__":
    app()
