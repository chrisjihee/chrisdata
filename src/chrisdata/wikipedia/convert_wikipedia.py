import json
import math
from typing import Iterable

import typer
from pydantic import Field

from chrisbase.data import InputOption, OutputOption, FileOption, TableOption, FileStreamer, MongoStreamer, IOArguments
from chrisbase.data import JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat, new_path, merge_dicts
from chrisbase.util import mute_tqdm_cls
from . import *

logger = logging.getLogger(__name__)
parsed_ids = set()  # for duplicated crawling data


class ExtraOption(BaseModel):
    export: bool = Field(default=True)
    processor: str | None = Field(default=None)


def convert_one(item: dict) -> WikipediaStat | None:
    doc: WikipediaCrawlResult = WikipediaCrawlResult.model_validate(item)
    doc.title = doc.title.strip() if doc.title else ""
    if not doc.page_id or doc.page_id in parsed_ids or not doc.title or not doc.section_list:
        return None
    sections = [str(x[-1]).strip() for x in doc.section_list]
    parsed_ids.add(doc.page_id)
    return WikipediaStat(title=doc.title, page_id=doc.page_id, length=sum([len(x) for x in sections]))


def convert_many(item: str | Iterable[str], args: IOArguments, writer: MongoStreamer, item_is_batch: bool = True):
    inputs = item if item_is_batch else [item]
    inputs = [json.loads(i) for i in inputs]
    outputs = [convert_one(i) for i in inputs]
    outputs = {v.id: v for v in outputs if v}
    records = [merge_dicts({"_id": k}, v.model_dump()) for k, v in outputs.items()]
    if args.env.debugging:
        logger.debug(f"convert_many: {len(inputs)} -> {len(outputs)} -> {len(records)}")
    if len(records) > 0:
        writer.table.insert_many(records)


@app.command()
def convert(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="convert_wikipedia"),
        logging_home: str = typer.Option(default="output/wikipedia/convert"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),  # TODO: Replace with False
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),  # TODO: Replace with -1 or 1410203
        input_batch: int = typer.Option(default=100),  # TODO: Replace with 100
        input_inter: int = typer.Option(default=50000),  # TODO: Replace with 50000
        input_file_home: str = typer.Option(default="input/wikipedia"),
        input_file_name: str = typer.Option(default="kowiki-20230701-all-titles-in-ns0-crawl.jsonl"),
        # output
        output_file_home: str = typer.Option(default="output/wikipedia"),
        output_file_name: str = typer.Option(default="wikipedia-20230701-convert.jsonl"),
        output_file_mode: str = typer.Option(default="w"),
        output_table_home: str = typer.Option(default="localhost:8800/wikipedia"),
        output_table_name: str = typer.Option(default="wikipedia-20230701-convert"),
        output_table_reset: bool = typer.Option(default=True),
        # option
        export: bool = typer.Option(default=True),
        processor: str = typer.Option(default="convert_many"),
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
        file=FileOption(
            home=input_file_home,
            name=input_file_name,
            required=True,
        ),
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
        export=export,
        processor=processor,
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
        option=extra_opt,
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
        input_data = args.input.ready_inputs(input_file, total=len(input_file))
        logger.info(f"Convert from [{input_file.opt}] to [{output_file.opt}, {output_table.opt}]")
        logger.info(f"- [input] total={args.input.total} | start={args.input.start} | limit={args.input.limit}"
                    f" | {type(input_data).__name__}={input_data.num_item}{f'x{args.input.batch}ea' if input_data.has_batch_items() else ''}")
        logger.info(f"- [output] file.reset={args.output.file.reset} | file.mode={args.output.file.mode}")
        logger.info(f"- [output] table.reset={args.output.table.reset} | table.timeout={args.output.table.timeout}")
        with tqdm(total=input_data.num_item, unit="item", pre="=>", desc="converting", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
            for item in input_data.items:
                convert_many(item=item, args=args, writer=output_table,
                             item_is_batch=input_data.has_batch_items())
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)

        if extra_opt.export:
            with tqdm(total=len(output_table), unit="row", pre="=>", desc="exporting", unit_divisor=args.input.inter * 100) as prog:
                for row in output_table:
                    row = WikipediaStat.model_validate(row)
                    output_file.fp.write(row.model_dump_json() + '\n')
                    prog.update()
                    if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                        logger.info(prog)
                logger.info(f"Export {prog.n}/{args.input.total} rows to [{output_file.opt}]")
