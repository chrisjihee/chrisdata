import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from qwikidata.json_dump import WikidataJsonDump

from chrisbase.data import FileStreamer, MongoStreamer
from chrisbase.data import IOArguments, InputOption, OutputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv, OptionData
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls
from chrisdata.wikidata import *

logger = logging.getLogger(__name__)


@dataclass
class FilterOption(OptionData):
    lang1: str = field(default="ko")
    lang2: str = field(default="en")
    strict: bool = field(default=False)
    truthy: bool = field(default=False)


@dataclass
class ParseArguments(IOArguments):
    filter: FilterOption | None = field(default=None)

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
            to_dataframe(columns=columns, raw=self.output, data_prefix="input", data_exclude=["file", "table", "index"]),
            to_dataframe(columns=columns, raw=self.output.file, data_prefix="output.file") if self.output.file else None,
            to_dataframe(columns=columns, raw=self.output.table, data_prefix="output.table") if self.output.table else None,
            to_dataframe(columns=columns, raw=self.output.index, data_prefix="output.index") if self.output.index else None,
            to_dataframe(columns=columns, raw=self.filter, data_prefix="filter"),
        ]).reset_index(drop=True)


def parse_one(x: dict, args: ParseArguments):
    if args.env.debugging:
        logger.info('')
        logger.info("*" * 120)
        logger.info(f"parse_one: {x['type']} {x['id']}")
        Path(f"debug-{x['id']}.json").write_text(json.dumps(x, ensure_ascii=False, indent=2))
        logger.info("-" * 120)

    def debug_return(r):
        if args.env.debugging:
            logger.info("*" * 120)
        return r

    lang1_code = LanguageCode(args.filter.lang1)
    lang2_code = LanguageCode(args.filter.lang2)
    row = WikidataUnit(
        _id=x['id'],
        ns=x['ns'],
        type=x['type'],
        time=x['modified'],
    )
    if row.type == "item" and row.ns == 0:
        item = WikidataItemEx(x)
        row.label1 = item.get_label(lang1_code)
        row.label2 = item.get_label(lang2_code)
        row.title1 = item.get_wiki_title(lang1_code)
        row.title2 = item.get_wiki_title(lang2_code)
        if args.filter.strict and (not row.label1 or not row.title1):
            return debug_return(None)
        row.alias1 = item.get_aliases(lang1_code)
        row.alias2 = item.get_aliases(lang2_code)
        row.descr1 = item.get_description(lang1_code)
        row.descr2 = item.get_description(lang2_code)
        row.claims = item.get_claims(args)
        return debug_return(row)
    elif row.type == "property":
        prop = WikidataPropertyEx(x)
        row.label1 = prop.get_label(lang1_code)
        row.label2 = prop.get_label(lang2_code)
        row.alias1 = prop.get_aliases(lang1_code)
        row.alias2 = prop.get_aliases(lang2_code)
        row.descr1 = prop.get_description(lang1_code)
        row.descr2 = prop.get_description(lang2_code)
        row.claims = prop.get_claims(args)
        return debug_return(row)
    elif row.type == "lexeme":
        lexm = WikidataLexemeEx(x)
        row.label1 = lexm.get_lemma(lang1_code)
        row.label2 = lexm.get_lemma(lang2_code)
        if args.filter.strict and not row.label1:
            return debug_return(None)
        row.descr1 = lexm.get_gloss(lang1_code)
        row.descr2 = lexm.get_gloss(lang2_code)
        row.claims = lexm.get_claims(args)
        return debug_return(row)
    return debug_return(None)


def parse_many(batch: Iterable[dict], wrapper: MongoStreamer, args: ParseArguments):
    rows = [parse_one(x, args) if wrapper.count({"_id": x['id']}) == 0 else None
            for x in batch]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        wrapper.table.insert_many(rows)


@app.command()
def parse(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="parse_wikidata"),
        output_home: str = typer.Option(default="output/parse_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=True),
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=1000),
        input_inter: int = typer.Option(default=10000),
        input_total: int = typer.Option(default=105485440),  # https://www.wikidata.org/wiki/Wikidata:Statistics
        input_file_home: str = typer.Option(default="input/Wikidata"),
        input_file_name: str = typer.Option(default="wikidata-20230911-all.json.bz2"),
        # output
        output_file_home: str = typer.Option(default="output/Wikidata"),
        output_file_name: str = typer.Option(default="wikidata-20230911-all-parse-ko-en.jsonl"),
        output_file_mode: str = typer.Option(default="w"),
        output_table_home: str = typer.Option(default="localhost:6382/Wikidata"),
        output_table_name: str = typer.Option(default="wikidata-20230911-all-parse-ko-en"),
        output_table_reset: bool = typer.Option(default=True),
        # filter
        filter_lang1: str = typer.Option(default="ko"),
        filter_lang2: str = typer.Option(default="en"),
        filter_strict: bool = typer.Option(default=False),
        filter_truthy: bool = typer.Option(default=False),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        output_home=output_home,
        logging_file=logging_file,
        msg_level=logging.INFO,  # if not debugging else logging.DEBUG,
        msg_format=LoggingFormat.CHECK_24,  # if not debugging else LoggingFormat.DEBUG_24,
    )
    input_opt = InputOption(
        start=input_start,
        limit=input_limit if not debugging else 1,
        batch=input_batch if not debugging else 2,
        inter=input_inter if not debugging else 1,
        file=FileOption(
            home=input_file_home,
            name=input_file_name,
        ) if input_file_home and input_file_name else None,
        table=TableOption(
            home=output_table_home,
            name=output_table_name,
            reset=output_table_reset,
        ) if output_table_home and output_table_name else None,
    )
    filter_opt = FilterOption(
        lang1=filter_lang1,
        lang2=filter_lang2,
        strict=filter_strict,
        truthy=filter_truthy,
    )
    args = ParseArguments(
        env=env,
        input=input_opt,
        filter=filter_opt,
    )
    tqdm = mute_tqdm_cls()
    save_file = (args.env.output_home / f"{output_table_name}-{args.env.time_stamp}.jsonl")
    assert args.input.file, "data.file is required"
    assert args.input.table, "data.table is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoStreamer(args.input.table) as data_table,
        FileStreamer(args.input.file) as data_file,
        save_file.open("w") as writer,
    ):
        # parse dump data
        inputs = args.input.ready_inputs(WikidataJsonDump(str(data_file.path)), input_total)
        logger.info(f"Parse from [{data_file.opt}] to [{data_table.opt}]")
        logger.info(f"- amount: inputs={input_total}, batches={inputs.total}")
        logger.info(f"- filter: lang1={args.filter.lang1}, lang2={args.filter.lang2}")
        progress, interval = (
            tqdm(inputs.batches, total=inputs.total, unit="batch", pre="*", desc="parsing"),
            math.ceil(args.input.inter / args.input.batch),
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            parse_many(batch=x, wrapper=data_table, args=args)
        logger.info(progress)

        # save parsed data
        progress, interval = (
            tqdm(data_table, total=len(data_table), unit="row", pre="*", desc="saving"),
            args.input.inter * 100,
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            writer.write(json.dumps(x, ensure_ascii=False, indent=None if not debugging else 2) + '\n')
        logger.info(progress)
        logger.info(f"Saved {len(data_table)} rows to [{save_file}]")
