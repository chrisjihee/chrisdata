from typing import Iterable

import typer

from chrisbase.data import FileStreamer
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat, new_path
from chrisbase.util import mute_tqdm_cls
from . import *

logger = logging.getLogger(__name__)


def filter_one(x: dict) -> str | None:
    if x['type'] != "item":
        return None
    if x['title1'] and len(x['title1']) > 0:
        return f"{x['_id']}\n"
    else:
        return None


def filter_many(item: dict | Iterable[dict], writer: FileStreamer, item_is_batch: bool = True):
    batch = item if item_is_batch else [item]
    rows = [filter_one(x) for x in batch]
    for row in rows:
        if row:
            writer.fp.write(row)


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
        input_batch: int = typer.Option(default=100),
        input_inter: int = typer.Option(default=10000),
        input_table_home: str = typer.Option(default="localhost:8800/wikidata"),
        input_table_name: str = typer.Option(default="wikidata-20240916-parse"),
        input_table_timeout: int = typer.Option(default=3600),
        # output
        output_file_home: str = typer.Option(default="output/wikidata"),
        output_file_name: str = typer.Option(default="wikidata-20240916-korean.txt"),
        output_file_mode: str = typer.Option(default="w"),
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
        table=TableOption(
            home=input_table_home,
            name=input_table_name,
            timeout=input_table_timeout * 1000,
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
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.table, "input.table is required"
    assert args.output.file, "output.file is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoStreamer(args.input.table) as input_table,
        FileStreamer(args.output.file) as output_file,
    ):
        # filter korean wikipedia entities
        input_data = args.input.ready_inputs(input_table, total=len(input_table))
        logger.info(f"Filter from [{input_table.opt}] to [{output_file.opt}]")
        logger.info(f"- [input] total={args.input.total} | start={args.input.start} | limit={args.input.limit} | table.timeout={args.input.table.timeout}"
                    f" | {type(input_data).__name__}={input_data.num_item}{f'x{args.input.batch}ea' if input_data.has_batch_items() else ''}")
        logger.info(f"- [output] file.reset={args.output.file.reset} | file.mode={args.output.file.mode}")
        with tqdm(total=input_data.num_item, unit="item", pre="=>", desc="filtering", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
            for item in input_data.items:
                filter_many(item=item, writer=output_file,
                            item_is_batch=input_data.has_batch_items())
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
