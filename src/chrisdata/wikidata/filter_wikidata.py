import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

import typer

from chrisbase.data import FileStreamer, MongoStreamer
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv, OptionData
from chrisbase.io import LoggingFormat, new_path
from chrisbase.util import mute_tqdm_cls
from chrisdata.wikidata import *

logger = logging.getLogger(__name__)


@dataclass
class ExtraOption(OptionData):
    processor: str = field(default="filter_many1")


def filter_one(x: dict) -> WikidataUnit | None:
    subject: WikidataUnit = WikidataUnit.from_dict(x)
    if subject.type != "item":
        return None
    if subject.title1 and len(subject.title1) > 0:
        return subject['_id']
    else:
        return None


def filter_many1(item: dict | Iterable[dict], args: IOArguments, writer: MongoStreamer, writer2: FileStreamer, item_is_batch: bool = True):
    batch = item if item_is_batch else [item]
    rows = [filter_one(x) for x in batch]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


def filter_many2(item: dict | Iterable[dict], args: IOArguments, writer: MongoStreamer, writer2: FileStreamer, item_is_batch: bool = True):
    batch = item if item_is_batch else [item]
    with ProcessPoolExecutor(max_workers=args.env.max_workers) as exe:
        jobs = [exe.submit(filter_one, x) for x in batch]
        rows = [job.result(timeout=args.env.waiting_sec) for job in jobs]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


def filter_many3(item: dict | Iterable[dict], args: IOArguments, writer: MongoStreamer, writer2: FileStreamer, item_is_batch: bool = True):
    batch = item if item_is_batch else [item]
    with multiprocessing.Pool(processes=args.env.max_workers) as pool:
        jobs = [pool.apply_async(filter_one, (x,)) for x in batch]
        rows = [job.get(timeout=args.env.waiting_sec) for job in jobs]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


@app.command()
def filter(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="filter_wikidata"),
        logging_home: str = typer.Option(default="output/wikidata/filter"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=1000),
        input_inter: int = typer.Option(default=5000),
        input_total: int = typer.Option(default=113850250),  # 112473850 vs. 113850250 # https://www.wikidata.org/wiki/Wikidata:Statistics  # TODO: Replace with (actual count)
        input_table_home: str = typer.Option(default="localhost:8800/wikidata"),
        input_table_name: str = typer.Option(default="wikidata-20240916-parse"),
        # output
        output_file_home: str = typer.Option(default="output/wikidata"),
        output_file_name: str = typer.Option(default="wikidata-20240916-filter.jsonl"),
        output_file_mode: str = typer.Option(default="w"),
        output_table_home: str = typer.Option(default="localhost:8800/wikidata"),
        output_table_name: str = typer.Option(default="wikidata-20240916-filter"),
        output_table_reset: bool = typer.Option(default=False),
        # option
        processor: str = typer.Option(default="filter_many1"),
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
        start=input_start if not debugging else 0,
        limit=input_limit if not debugging else 2,
        batch=input_batch if not debugging else 1,
        inter=input_inter if not debugging else 1,
        total=input_total,
        table=TableOption(
            home=input_table_home,
            name=input_table_name,
            required=True,
        )
    )
    output_opt = OutputOption(
        file=FileOption(
            home=output_file_home,
            name=new_path(output_file_name, post=env.time_stamp),
            mode=output_file_mode,
            required=True,
        ),
        table=TableOption(
            home=output_table_home,
            name=output_table_name,
            reset=output_table_reset,
            required=True,
        )
    )
    extra_opt = ExtraOption(
        processor=processor if env.max_workers > 1 else "filter_many1",
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
        option=extra_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.table, "input.table is required"
    assert args.output.file, "output.file is required"
    assert args.output.table, "output.table is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoStreamer(args.input.table) as input_table,
        FileStreamer(args.output.file) as output_file,
        MongoStreamer(args.output.table) as output_table,
    ):
        # filter korean wikipedia entities
        input_data = args.input.ready_inputs(input_table, total=len(input_table))
        logger.info(f"Filter from [{input_table.opt}] to [{output_table.opt}]")
        logger.info(f"- [input] total={args.input.total} | start={args.input.start} | limit={args.input.limit}"
                    f" | {type(input_data).__name__}={input_data.num_item}{f'x{args.input.batch}ea' if input_data.has_batch_items() else ''}")
        logger.info(f"- [output] table.reset={args.output.table.reset} | table.timeout={args.output.table.timeout}")
        with tqdm(total=input_data.num_item, unit="item", pre="=>", desc="filtering", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
            for item in input_data.items:
                if args.option.processor == "filter_many1":
                    filter_many1(item=item, args=args, writer=output_table, item_is_batch=input_data.has_batch_items())
                elif args.option.processor == "filter_many2":
                    filter_many2(item=item, args=args, writer=output_table, item_is_batch=input_data.has_batch_items())
                elif args.option.processor == "filter_many3":
                    filter_many3(item=item, args=args, writer=output_table, item_is_batch=input_data.has_batch_items())
                else:
                    assert False, f"Unknown processor: {args.option.processor}"
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
