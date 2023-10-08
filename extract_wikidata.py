import logging
import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Iterable

import pandas as pd
import typer

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, CommonArguments, TypedData, InputChannel, OutputChannel, FileRewriter
from chrisbase.data import InputOption, OutputOption, TableOption
from chrisbase.data import MongoRewriter
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls

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
    # descr1: str
    # descr2: str


@dataclass
class SingleTriple(TypedData):
    entity1: EntityInWiki
    entity2: EntityInWiki
    relation: Relation
    hits: int = field(default=0)
    pmi: float = field(default=0.0)
    score: float = field(default=0.0)

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
class DoubleTriple(TypedData):
    entity1: EntityInWiki
    entity2: EntityInWiki
    entity3: EntityInWiki
    relation1: Relation
    relation2: Relation
    hits1: int = field(default=0)
    hits2: int = field(default=0)
    pmi1: float = field(default=0.0)
    pmi2: float = field(default=0.0)
    score1: float = field(default=0.0)
    score2: float = field(default=0.0)

    @staticmethod
    def from_triples(
            triple1: "SingleTriple",
            triple2: "SingleTriple"
    ) -> Optional["DoubleTriple"]:
        if (triple2.entity2.entity != triple1.entity1.entity
                and triple2.relation.id != triple1.relation.id):
            return DoubleTriple(
                entity1=triple1.entity1,
                entity2=triple1.entity2,
                entity3=triple2.entity2,
                relation1=triple1.relation,
                relation2=triple2.relation,
                hits1=triple1.hits,
                hits2=triple2.hits,
                score1=triple1.score,
                score2=triple2.score,
                pmi1=triple1.pmi,
                pmi2=triple2.pmi,
            )
        else:
            return None


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
            to_dataframe(columns=columns, raw=self.output.table, data_prefix="output.table"),
        ]).reset_index(drop=True)


def extract_one(x: dict, input_table: MongoRewriter):
    single1 = SingleTriple.from_dict(x)
    if single1.entity1.entity != single1.entity2.entity:
        bridge = single1.entity2
        for y in input_table.table.find({"entity1.entity": bridge.entity}):
            single2 = SingleTriple.from_dict(y)
            double = DoubleTriple.from_triples(single1, single2)
            if double:
                yield double


def extract_many(batch: Iterable[dict], wrapper: MongoRewriter | FileRewriter, input_table: MongoRewriter):
    batch_units = [extract_one(x, input_table) for x in batch]
    all_units = [x for batch in batch_units for x in batch]
    rows = [row.to_dict() for row in all_units if row]
    if len(rows) > 0:
        wrapper.table.insert_many(rows)


@app.command()
def extract(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="extract_wikidata"),
        output_home: str = typer.Option(default="output-extract_wikidata"),
        logging_file: str = typer.Option(default="extract.out"),
        debugging: bool = typer.Option(default=False),
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=10),
        input_inter: int = typer.Option(default=100),
        input_total: int = typer.Option(default=-1),
        input_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        input_table_name: str = typer.Option(default="wikidata-20230920-search-kowiki"),
        input_table_sort: Tuple[str, int] = typer.Option(default=("entity2.entity", 1)),
        # output
        output_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        output_table_name: str = typer.Option(default="wikidata-20230920-extract-kowiki"),
        output_table_reset: bool = typer.Option(default=False),
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
        table=TableOption(
            home=output_table_home,
            name=output_table_name,
            reset=output_table_reset,
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
    assert args.output.table, "output.table is required"

    with (
        JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoRewriter(args.input.table) as input_table,
        MongoRewriter(args.output.table) as output_table,
    ):
        # extract connected triple pairs
        inp: InputChannel = args.input.first_usable(input_table, total=len(input_table))
        out: OutputChannel = args.output.first_usable(output_table)
        logger.info(f"Extract from [{inp.wrapper.opt}] to [{out.rewriter.opt}]")
        logger.info(f"- amount: inputs={inp.num_input}, batches={inp.total}")
        progress, interval = (
            tqdm(inp.batches, total=inp.total, unit="batch", pre="*", desc="parsing"),
            math.ceil(args.input.inter / args.input.batch),
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            extract_many(batch=x, wrapper=out.rewriter, input_table=input_table)
        logger.info(progress)


if __name__ == "__main__":
    app()
