import json
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

import typer

from chrisbase.data import FileStreamer, MongoStreamer
from chrisbase.data import IOArguments, InputOption, OutputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat, new_path
from chrisbase.util import mute_tqdm_cls
from chrisdata.net import app, IPCheckResult

logger = logging.getLogger(__name__)
check_target = "https://api64.ipify.org?format=json"


def process_one(idx: int, args: IOArguments):
    if args.env.calling_sec > 0:
        time.sleep(args.env.calling_sec)
    http_client = args.env.http_clients[idx]
    local_addr = args.env.http_clients.get_local_addr(idx)
    _id = '.'.join(f"{int(a):03d}" for a in local_addr.split('.')[-2:])

    response = http_client.get(check_target)
    result = IPCheckResult(
        _id=_id,
        uri=check_target,
        ip=local_addr,
        status=response.status_code,
        elapsed=round(response.elapsed.total_seconds(), 3),
        size=round(response.num_bytes_downloaded / 1024, 6),
        text=response.text,
    )
    return result


def process_many1(batch: Iterable[int], args: IOArguments, writer: MongoStreamer):
    rows = [process_one(x, args) for x in batch]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


def process_many2(batch: Iterable[int], args: IOArguments, writer: MongoStreamer):
    with ProcessPoolExecutor(max_workers=args.env.max_workers) as exe:
        jobs = [exe.submit(process_one, x, args) for x in batch]
        rows = [job.result(timeout=args.env.waiting_sec) for job in jobs]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


@app.command()
def check(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="check_ip_addrs"),
        output_home: str = typer.Option(default="output/check_ip_addrs"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=5),
        debugging: bool = typer.Option(default=False),
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=10),
        input_inter: int = typer.Option(default=1),
        # output
        output_file_home: str = typer.Option(default="output/check_ip_addrs"),
        output_file_name: str = typer.Option(default="check_ip_addrs.jsonl"),
        output_file_mode: str = typer.Option(default="w"),
        output_table_home: str = typer.Option(default="localhost:8800/device"),
        output_table_name: str = typer.Option(default="check_ip_addrs"),
        output_table_reset: bool = typer.Option(default=True),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        logging_home=output_home,
        logging_file=logging_file,
        message_level=logging.DEBUG if debugging else logging.INFO,
        message_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
        max_workers=1 if debugging else max(max_workers, 1),
    )
    assert len(env.http_clients) > 0, f"env.http_clients={env.http_clients}"
    input_opt = InputOption(
        start=input_start,
        limit=input_limit,
        batch=input_batch,
        inter=input_inter,
        data=list(range(len(env.http_clients))),
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
        ),
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.data, "input.data is required"
    assert args.output.file, "output.file is required"
    assert args.output.table, "output.table is required"
    logging.getLogger("httpx").setLevel(logging.WARNING)

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(args.output.file) as output_file,
        MongoStreamer(args.output.table) as output_table,
    ):
        # check local ip addresses
        input_total = len(args.env.http_clients)
        input_data = args.input.ready_inputs(args.input.data, input_total)
        logger.info(f"Check {input_total} addresses to [{output_table.opt}]")
        logger.info(f"- amount: {args.input.total}{'' if input_data.has_single_items() else f' * {args.input.batch}'} ({type(input_data).__name__})")
        with tqdm(total=input_data.num_item, unit="item", pre="=>", desc="checking", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
            for batch in input_data.items:
                if args.env.max_workers <= 1:
                    process_many1(batch=batch, args=args, writer=output_table)
                else:
                    process_many2(batch=batch, args=args, writer=output_table)
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)

        # export results
        with tqdm(total=len(output_table), unit="row", pre="=>", desc="exporting", unit_divisor=args.input.inter * 10) as prog:
            for row in output_table:
                output_file.fp.write(json.dumps(row, ensure_ascii=False) + '\n')
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
            logger.info(f"Export {prog.n}/{input_total} rows to [{output_file.opt}]")
