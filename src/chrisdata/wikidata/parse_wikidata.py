import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from qwikidata.json_dump import WikidataJsonDump

from chrisbase.data import FileStreamer, MongoStreamer
from chrisbase.data import IOArguments, InputOption, OutputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv, OptionData
from chrisbase.io import LoggingFormat, new_path
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


def parse_many1(batch: Iterable[dict], args: ParseArguments, writer: MongoStreamer):
    if not writer.opt.reset:
        batch = [x for x in batch if writer.count({"_id": x['id']}) == 0]
    rows = [parse_one(x, args) for x in batch]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


def parse_many2(batch: Iterable[dict], args: ParseArguments, writer: MongoStreamer):
    if not writer.opt.reset:
        batch = [x for x in batch if writer.count({"_id": x['id']}) == 0]
    with ProcessPoolExecutor(max_workers=args.env.max_workers) as exe:
        jobs = [exe.submit(parse_one, x, args) for x in batch]
        rows = [job.result(timeout=args.env.waiting_sec) for job in jobs]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


def parse_many3(batch: Iterable[dict], args: ParseArguments, writer: MongoStreamer):
    if not writer.opt.reset:
        batch = [x for x in batch if writer.count({"_id": x['id']}) == 0]
    with multiprocessing.Pool(processes=args.env.max_workers) as pool:
        jobs = [pool.apply_async(parse_one, (x, args)) for x in batch]
        rows = [job.get(timeout=args.env.waiting_sec) for job in jobs]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


@app.command()
def parse(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="parse_wikidata"),
        output_home: str = typer.Option(default="output/parse_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),  # TODO: change to False
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),  # TODO: change to -1
        input_batch: int = typer.Option(default=1000),  # TODO: change to 1000
        input_inter: int = typer.Option(default=10000),  # TODO: change to 10000
        input_total: int = typer.Option(default=105485440),  # https://www.wikidata.org/wiki/Wikidata:Statistics
        input_file_home: str = typer.Option(default="input/Wikidata"),
        input_file_name: str = typer.Option(default="wikidata-20230911-all.json.bz2"),
        # output
        output_file_home: str = typer.Option(default="output/Wikidata"),
        output_file_name: str = typer.Option(default="wikidata-20230911-all-parse-ko-en.jsonl"),
        output_file_mode: str = typer.Option(default="w"),
        output_table_home: str = typer.Option(default="localhost:6382/Wikidata"),
        output_table_name: str = typer.Option(default="wikidata-20230911-all-parse-ko-en"),
        output_table_reset: bool = typer.Option(default=False),
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
        msg_format=LoggingFormat.CHECK_36,  # if not debugging else LoggingFormat.DEBUG_24,
        max_workers=1 if debugging else max(max_workers, 1),
    )
    input_opt = InputOption(
        start=input_start,
        limit=input_limit if not debugging else 1,
        batch=input_batch if not debugging else 2,
        inter=input_inter if not debugging else 1,
        file=FileOption(
            home=input_file_home,
            name=input_file_name,
            strict=True,
        ),
    )
    output_opt = OutputOption(
        file=FileOption(
            home=output_file_home,
            name=new_path(output_file_name, post=env.time_stamp),
            mode=output_file_mode,
            strict=True,
        ),
        table=TableOption(
            home=output_table_home,
            name=output_table_name,
            reset=output_table_reset,
            strict=True,
        )
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
        output=output_opt,
        filter=filter_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.file, "input.file is required"
    assert args.output.file, "output.file is required"
    assert args.output.table, "output.table is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(args.input.file) as input_file,
        FileStreamer(args.output.file) as output_file,
        MongoStreamer(args.output.table) as output_table,
    ):
        # parse dump data
        input_items = args.input.ready_inputs(WikidataJsonDump(str(input_file.path)), input_total)
        logger.info(f"Parse from [{input_file.opt}] to [{output_table.opt}]")
        logger.info(f"- amount: {input_items.total}{'' if input_items.has_single_items() else f' * {args.input.batch}'} ({type(input_items).__name__})")
        logger.info(f"- filter: lang1={args.filter.lang1}, lang2={args.filter.lang2}, strict={args.filter.strict}, truthy={args.filter.truthy}")
        with tqdm(total=input_items.total, unit="item", pre="=>", desc="parsing", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
            for batch in input_items.items:
                if args.env.max_workers <= 1:
                    parse_many1(batch=batch, args=args, writer=output_table)
                else:
                    parse_many3(batch=batch, args=args, writer=output_table)
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)

        # export parsed data
        with tqdm(total=len(output_table), unit="row", pre="=>", desc="exporting", unit_divisor=args.input.inter * 100) as prog:
            for row in output_table:
                output_file.fp.write(json.dumps(row, ensure_ascii=False, indent=None if not debugging else 2) + '\n')
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
            logger.info(f"Export {prog.n}/{input_total} rows to [{output_file.opt}]")