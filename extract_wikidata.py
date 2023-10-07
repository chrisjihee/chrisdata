import json
import logging
import math
import re
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Iterable, Tuple

import bson.json_util
import pandas as pd
import typer

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, CommonArguments, OptionData, TypedData
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption, IndexOption
from chrisbase.data import LineFileWrapper, MongoDBWrapper, ElasticSearchWrapper
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls
from parse_wikidata import WikidataUnit

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class EntityInWiki(TypedData):
    entity: str
    hits: int
    score: float


@dataclass
class Relation(TypedData):
    id: str
    label1: str
    label2: str
    descr1: str
    descr2: str


@dataclass
class TripleInWiki(TypedData):
    entity1: EntityInWiki
    entity2: EntityInWiki
    relation: Relation
    hits: int
    score: float = field(default=0.0)
    pmi: float = field(default=0.0)

    @staticmethod
    def calc_pmi(h_xy: float, h_x: float, h_y: float, n: int = 10000, e: float = 0.0000001) -> float:
        p_xy = h_xy / n
        p_x = h_x / n
        p_y = h_y / n
        return math.log2((p_xy + e) / ((p_x * p_y) + e))

    def __post_init__(self):
        if self.score is None:
            self.score = 0.0
        self.pmi = self.calc_pmi(self.hits, self.entity1.hits, self.entity2.hits)


@dataclass
class ExtractArguments(CommonArguments):
    input: InputOption = field()
    output: OutputOption = field()

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.input, data_prefix="input", data_exclude=["file", "table", "index"]),
            to_dataframe(columns=columns, raw=self.input.table, data_prefix="input.table"),
            to_dataframe(columns=columns, raw=self.output.file, data_prefix="output.file"),
        ]).reset_index(drop=True)


@app.command()
def extract(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="search_wikidata"),
        output_home: str = typer.Option(default="output-search_wikidata"),
        logging_file: str = typer.Option(default="extract.out"),
        debugging: bool = typer.Option(default=False),
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=1000),
        input_inter: int = typer.Option(default=5000),
        input_total: int = typer.Option(default=-1),
        input_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        input_table_name: str = typer.Option(default="wikidata-20230920-search-kowiki"),
        input_table_sort: Tuple[str, int] = typer.Option(default=("entity2.entity", 1)),
        # output
        output_file_home: str = typer.Option(default="output-search_wikidata"),
        output_file_name: str = typer.Option(default="wikidata-20230920-search-kowiki-new.jsonl"),
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
    input_opt = InputOption(
        start=input_start,
        limit=input_limit,
        batch=input_batch,
        inter=input_inter,
        total=input_total,
        table=TableOption(
            home=input_table_home,
            name=input_table_name,
            sort=[input_table_sort],
            strict=True,
        ),
    )
    output_opt = OutputOption(
        file=FileOption(
            home=output_file_home,
            name=output_file_name,
            mode="w",
            strict=True,
        ),
    )
    args = ExtractArguments(
        env=env,
        input=input_opt,
        output=output_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.table, "input.table is required"
    assert args.output.file, "output.file is required"

    with (
        JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoDBWrapper(args.input.table) as input_table,
        LineFileWrapper(args.output.file) as output_file,
    ):
        # extract connected triple pairs
        args.input.total = len(input_table)
        # for x in islice(input_table.table.find().sort("entity2.entity"), 10):
        #     print(f"x={x}")
        # exit(1)
        inputs = args.input.select_inputs(input_table)
        outputs = args.output.select_outputs(output_file)
        logger.info(f"Extract from [{inputs.wrapper.opt}] to [{outputs.wrapper.opt}]")
        logger.info(f"- amount: inputs={inputs.num_input}, batches={inputs.num_batch}")
        rows, num_row = input_table, len(input_table)
        progress, interval = (
            tqdm(rows, total=num_row, unit="row", pre="*", desc="saving"),
            args.input.inter * 100,
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            triple1 = TripleInWiki.from_dict(x)
            print(triple1.entity1)
            print(triple1.entity2)
            print(triple1.relation)
            for y in input_table.table.find({"entity1": triple1.entity2.to_dict()}):
                triple2 = TripleInWiki.from_dict(y)
                print(triple2.entity1)
                print(triple2.entity2)
                print(triple2.relation)
            exit(1)
        logger.info(progress)
        logger.info(f"Saved {num_row} rows to [{output_file.path}]")


if __name__ == "__main__":
    app()
