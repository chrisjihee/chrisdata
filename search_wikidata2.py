import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
import re

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, CommonArguments, OptionData
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption, IndexOption
from chrisbase.data import LineFileWrapper, MongoDBWrapper, ElasticSearchWrapper
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls
from parse_wikidata import WikidataUnit

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class FilterOption(OptionData):
    min_char: int = field(default=2)
    max_char: int = field(default=20)
    max_word: int = field(default=5)
    black_prop: str | Path = field(default="black_prop.txt")
    num_black_prop: int = 0
    set_black_prop = set()
    parenth_ending = re.compile(r" \(.+?\)$")
    korean_starting = re.compile(r"^[가-힣].+")

    def __post_init__(self):
        self.black_prop = Path(self.black_prop)
        if self.black_prop.exists() and self.black_prop.is_file():
            lines = (x.strip() for x in self.black_prop.read_text().splitlines())
            self.set_black_prop = {x for x in lines if x}
            self.num_black_prop = len(self.set_black_prop)

    def invalid_prop(self, x: str) -> bool:
        return x in self.set_black_prop

    def normalize_title(self, x: str) -> str:
        return self.parenth_ending.sub("", x).strip()

    def invalid_title(self, full: str, norm: str) -> bool:
        return (
                len(norm) < self.min_char or
                len(norm) > self.max_char or
                len(norm.split()) > self.max_word or
                not self.korean_starting.match(norm) or
                norm.startswith("위키백과:") or norm.startswith("분류:") or norm.startswith("포털:") or
                norm.startswith("틀:") or norm.startswith("모듈:") or norm.startswith("위키프로젝트:") or
                full.endswith("(동음이의)")
        )


@dataclass
class SearchArguments(CommonArguments):
    input: InputOption = field()
    output: OutputOption = field()
    filter: FilterOption = field()

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.input, data_prefix="input", data_exclude=["file", "table", "index"]),
            to_dataframe(columns=columns, raw=self.input.file, data_prefix="input.file") if self.input.file else None,
            to_dataframe(columns=columns, raw=self.input.table, data_prefix="input.table") if self.input.table else None,
            to_dataframe(columns=columns, raw=self.input.index, data_prefix="input.index") if self.input.index else None,
            to_dataframe(columns=columns, raw=self.output.file, data_prefix="output.file") if self.output.file else None,
            to_dataframe(columns=columns, raw=self.output.table, data_prefix="output.table") if self.output.table else None,
            to_dataframe(columns=columns, raw=self.filter, data_prefix="filter"),
        ]).reset_index(drop=True)


def search_one(x: dict, input_table: MongoDBWrapper, input_index: ElasticSearchWrapper, opt: FilterOption):
    row = WikidataUnit.from_dict(x)
    if row.type == "item":
        subject_full = row.title1
        subject_norm = opt.normalize_title(subject_full)
        if not opt.invalid_title(full=subject_full, norm=subject_norm):
            print(f"subject={subject_norm}")
            for claim in row.claims:
                prop_id = claim['property']
                if not opt.invalid_prop(prop_id):
                    prop_res = input_table.table.find_one({'_id': prop_id})
                    prop_label = f"{prop_id}[{prop_res['label2'].replace('(', '').replace(')', '')}]"
                    value = claim['datavalue']
                    value_type = value['type']
                    if value_type == "wikibase-entityid":
                        entity_type = value['value']['entity-type']
                        if entity_type == "item":
                            entity_id = value['value']['id']
                            item_res = input_table.table.find_one({'_id': entity_id})
                            if item_res:
                                object_full = item_res['title1']
                                object_norm = opt.normalize_title(object_full)
                                if not opt.invalid_title(full=object_full, norm=object_norm):
                                    print(f"- {prop_label:100s} ====> \t\t\t{entity_id:10s} -> {object_norm}")
        print()
    return None


def search_many(batch: Iterable[dict], input_table: MongoDBWrapper, input_index: ElasticSearchWrapper, filter_opt: FilterOption):
    batch_units = [x for x in [search_one(x, input_table, input_index, opt=filter_opt) for x in batch] if x]
    print(f"len(batch_units)={len(batch_units)}")
    # for a in x:
    #     a = WikidataUnit.from_dict(a)
    #     # print("a", type(a), a)


@app.command()
def search(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="search_wikidata"),
        output_home: str = typer.Option(default="output-search_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # input
        input_start: int = typer.Option(default=11000),
        # input_limit: int = typer.Option(default=-1),
        input_limit: int = typer.Option(default=20000),
        input_batch: int = typer.Option(default=1000),
        input_inter: int = typer.Option(default=5000),
        input_total: int = typer.Option(default=1018174),
        input_file_home: str = typer.Option(default="input/wikimedia"),
        input_file_name: str = typer.Option(default="wikidata-20230920-parse-kowiki.jsonl"),
        input_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        input_table_name: str = typer.Option(default="wikidata-20230920-parse-kowiki"),
        input_index_home: str = typer.Option(default="localhost:9810"),
        input_index_name: str = typer.Option(default="wikipedia-20230920-index-kowiki"),
        input_index_user: str = typer.Option(default="elastic"),
        input_index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
        # output
        output_file_home: str = typer.Option(default="input/wikimedia"),
        output_file_name: str = typer.Option(default="wikidata-20230920-search-kowiki-new.jsonl"),
        output_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        output_table_name: str = typer.Option(default="wikidata-20230920-search-kowiki"),
        output_table_reset: bool = typer.Option(default=True),
        # filter
        filter_min_char: int = typer.Option(default=2),
        filter_max_char: int = typer.Option(default=20),
        filter_max_word: int = typer.Option(default=5),
        filter_black_prop: str = typer.Option(default="input/wikimedia/wikidata-black_prop.txt"),
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
        file=FileOption(
            home=input_file_home,
            name=input_file_name,
        ) if input_file_home and input_file_name else None,
        table=TableOption(
            home=input_table_home,
            name=input_table_name,
        ) if input_table_home and input_table_name else None,
        index=IndexOption(
            home=input_index_home,
            user=input_index_user,
            pswd=input_index_pswd,
            name=input_index_name,
        ) if input_index_home and input_index_name else None,
    )
    output_opt = OutputOption(
        file=FileOption(
            home=output_file_home,
            name=output_file_name,
            mode="w",
        ) if output_file_home and output_file_name else None,
        table=TableOption(
            home=output_table_home,
            name=output_table_name,
            reset=output_table_reset,
        ) if output_table_home and output_table_name else None,
    )
    filter_opt = FilterOption(
        min_char=filter_min_char,
        max_char=filter_max_char,
        max_word=filter_max_word,
        black_prop=filter_black_prop,
    )
    args = SearchArguments(
        env=env,
        input=input_opt,
        output=output_opt,
        filter=filter_opt,
    )
    tqdm = mute_tqdm_cls()
    logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
    assert args.input.index, "input.index is required"
    assert args.input.file or args.input.table, "input.file or input.table is required"
    assert args.output.file, "output.file is required"
    assert args.output.table, "output.table is required"

    with (
        JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoDBWrapper(args.output.table) as output_table, LineFileWrapper(args.output.file) as output_file,
        MongoDBWrapper(args.input.table) as input_table, LineFileWrapper(args.input.file) as input_file,
        ElasticSearchWrapper(args.input.index) as input_index,
    ):
        # search parsed data
        inputs = args.input.select_inputs(input_table, input_file)
        outputs = args.output.select_outputs(output_table, output_file)
        logger.info(f"Search from [{inputs.wrapper.opt}] with [{args.input.index}] to [{outputs.wrapper.opt}]")
        logger.info(f"- amount: inputs={inputs.num_input}, batches={inputs.num_batch}")
        logger.info(f"- filter: num_black_sect={args.filter.num_black_prop}, min_char={args.filter.min_char}, max_char={args.filter.max_char}, max_word={args.filter.max_word}")
        progress, interval = (
            tqdm(inputs.batches, total=inputs.num_batch, unit="batch", pre="*", desc="searching"),
            math.ceil(args.input.inter / args.input.batch)
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            search_many(batch=x, input_table=input_table, input_index=input_index, filter_opt=args.filter)
        logger.info(progress)
        # logger.info(f"Indexed {len(data_index)} documents to [{args.data.index}]")


if __name__ == "__main__":
    app()
