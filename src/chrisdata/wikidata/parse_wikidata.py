import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import typer
from qwikidata.json_dump import WikidataJsonDump

from chrisbase.data import FileStreamer, MongoStreamer
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv, OptionData
from chrisbase.io import LoggingFormat, new_path
from chrisbase.util import mute_tqdm_cls
from chrisdata.wikidata import *

logger = logging.getLogger(__name__)


@dataclass
class ParseWikidataOption(OptionData):
    processor: str = field(default="parse_many1")
    lang1: str = field(default="ko")
    lang2: str = field(default="en")
    strict: bool = field(default=False)
    truthy: bool = field(default=False)


def parse_one(x: dict, args: IOArguments):
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

    lang1_code = LanguageCode(args.option.lang1)
    lang2_code = LanguageCode(args.option.lang2)
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
        if args.option.strict and (not row.label1 or not row.title1):
            return debug_return(None)
        row.alias1 = item.get_aliases(lang1_code)
        row.alias2 = item.get_aliases(lang2_code)
        row.descr1 = item.get_description(lang1_code)
        row.descr2 = item.get_description(lang2_code)
        try:
            row.claims = item.get_claims(args)
            return debug_return(row)
        except Exception as e:
            logger.error(f"Error on parse_one(id={x['id']}, ns={x['ns']}, type={x['type']}): [{type(e).__name__}] {e}")
            return debug_return(None)
    elif row.type == "property":
        prop = WikidataPropertyEx(x)
        row.label1 = prop.get_label(lang1_code)
        row.label2 = prop.get_label(lang2_code)
        row.alias1 = prop.get_aliases(lang1_code)
        row.alias2 = prop.get_aliases(lang2_code)
        row.descr1 = prop.get_description(lang1_code)
        row.descr2 = prop.get_description(lang2_code)
        try:
            row.claims = prop.get_claims(args)
            return debug_return(row)
        except Exception as e:
            logger.error(f"Error on parse_one(id={x['id']}, ns={x['ns']}, type={x['type']}): [{type(e).__name__}] {e}")
            return debug_return(None)
    elif row.type == "lexeme":
        lexm = WikidataLexemeEx(x)
        row.label1 = lexm.get_lemma(lang1_code)
        row.label2 = lexm.get_lemma(lang2_code)
        if args.option.strict and not row.label1:
            return debug_return(None)
        row.descr1 = lexm.get_gloss(lang1_code)
        row.descr2 = lexm.get_gloss(lang2_code)
        try:
            row.claims = lexm.get_claims(args)
            return debug_return(row)
        except Exception as e:
            logger.error(f"Error on parse_one(id={x['id']}, ns={x['ns']}, type={x['type']}): [{type(e).__name__}] {e}")
            return debug_return(None)
    return debug_return(None)


def parse_many1(batch: Iterable[dict], args: IOArguments, writer: MongoStreamer):
    if not writer.opt.reset:
        batch = [x for x in batch if writer.count({"_id": x['id']}) == 0]
    rows = [parse_one(x, args) for x in batch]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


def parse_many2(batch: Iterable[dict], args: IOArguments, writer: MongoStreamer):
    if not writer.opt.reset:
        batch = [x for x in batch if writer.count({"_id": x['id']}) == 0]
    with ProcessPoolExecutor(max_workers=args.env.max_workers) as exe:
        jobs = [exe.submit(parse_one, x, args) for x in batch]
        rows = [job.result(timeout=args.env.waiting_sec) for job in jobs]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


def parse_many3(batch: Iterable[dict], args: IOArguments, writer: MongoStreamer):
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
        logging_home: str = typer.Option(default="output/parse_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),
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
        output_table_home: str = typer.Option(default="localhost:8800/Wikidata"),
        output_table_name: str = typer.Option(default="wikidata-20230911-all-parse-ko-en"),
        output_table_reset: bool = typer.Option(default=False),
        # option
        processor: str = typer.Option(default="parse_many1"),
        lang1: str = typer.Option(default="ko"),
        lang2: str = typer.Option(default="en"),
        strict: bool = typer.Option(default=False),
        truthy: bool = typer.Option(default=False),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        logging_home=logging_home,
        logging_file=logging_file,
        message_level=logging.INFO,
        message_format=LoggingFormat.CHECK_00,  # if not debugging else LoggingFormat.DEBUG_36,
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
    parse_opt = ParseWikidataOption(
        processor=processor,
        lang1=lang1,
        lang2=lang2,
        strict=strict,
        truthy=truthy,
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
        option=parse_opt,
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
        logger.info(f"- amount: total={input_total}, items={input_items.total}{'' if input_items.has_single_items() else f' * {args.input.batch}'} ({type(input_items).__name__})")
        logger.info(f"- option: lang1={args.option.lang1}, lang2={args.option.lang2}, strict={args.option.strict}, truthy={args.option.truthy}")
        with tqdm(total=input_items.total, unit="item", pre="=>", desc="parsing", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
            for batch in input_items.items:
                if args.option.processor == "parse_many1" or args.env.max_workers <= 1:
                    parse_many1(batch=batch, args=args, writer=output_table)
                elif args.option.processor == "parse_many2":
                    parse_many2(batch=batch, args=args, writer=output_table)
                elif args.option.processor == "parse_many3":
                    parse_many3(batch=batch, args=args, writer=output_table)
                else:
                    assert False, f"Unknown processor: {args.option.processor}"
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
