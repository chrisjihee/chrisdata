import json
import logging
import math
import time
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from typing import List, Iterable

import httpx
import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from more_itertools import ichunked

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments, TableOption, MongoDBTable
from chrisbase.io import LoggingFormat
from chrisbase.util import MongoDB, to_dataframe, mute_tqdm_cls, wait_future_jobs, terminate_processes

mongos: List[MongoDB] = []
logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class DataOption(OptionData):
    items: enumerate = field()
    total: int = field(default=0)
    limit: int = field(default=-1)
    batch: int = field(default=1)
    prog_interval: int = field(default=1)


@dataclass
class NetOption(OptionData):
    calling_sec: float = field(default=0.001),
    waiting_sec: float = field(default=300.0),


@dataclass
class ProgramArguments(CommonArguments):
    data: DataOption = field()
    table: TableOption = field()
    net: NetOption | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.net, data_prefix="net") if self.net else None,
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
            to_dataframe(columns=columns, raw=self.table, data_prefix="table"),
        ]).reset_index(drop=True)


@dataclass
class ProcessResult(DataClassJsonMixin):
    _id: str
    uri: str
    local_address: str
    status: int | None = None
    elapsed: float | None = None
    size: float | None = None
    text: str | None = None


def process_one(x: str, args: ProgramArguments):
    if args.net and args.net.calling_sec > 0:
        time.sleep(args.net.calling_sec)
    remote_page = "https://api64.ipify.org?format=json"
    local_address = x
    _id = '.'.join(local_address.split('.')[-2:])
    with httpx.Client(
            transport=httpx.HTTPTransport(local_address=local_address),
            timeout=httpx.Timeout(timeout=120.0)
    ) as cli:
        uri = remote_page
        response = cli.get(uri)
        result = ProcessResult(
            _id=_id,
            uri=uri,
            local_address=local_address,
            status=response.status_code,
            elapsed=round(response.elapsed.total_seconds(), 3),
            size=round(response.num_bytes_downloaded / 1024, 6),
            text=response.text,
        )
        return result


def process_many(batch: Iterable[str], args: ProgramArguments):
    rows = [process_one(x, args) for x in batch]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        with MongoDBTable(args.table) as table:
            table.insert_many(rows)


@app.command()
def check(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="check_ip_addrs"),
        output_home: str = typer.Option(default="output-check_ip_addrs"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        max_workers: int = typer.Option(default=10),
        # net
        calling_sec: float = typer.Option(default=0),
        waiting_sec: float = typer.Option(default=300.0),
        # data
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=5),
        prog_interval: int = typer.Option(default=1),
        # table
        db_host: str = typer.Option(default="localhost:27017"),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        output_home=output_home,
        logging_file=logging_file,
        msg_level=logging.DEBUG if debugging else logging.INFO,
        msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
        max_workers=1 if debugging else max(max_workers, 1),
    )
    args = ProgramArguments(
        env=env,
        net=NetOption(
            calling_sec=calling_sec,
            waiting_sec=waiting_sec,
        ),
        data=DataOption(
            items=islice(env.ip_addrs, env.num_ip_addrs),
            total=env.num_ip_addrs,
            limit=input_limit,
            batch=input_batch,
            prog_interval=prog_interval,
        ),
        table=TableOption(
            db_host=db_host,
            db_name=env.project,
            tab_name=env.job_name,
        ),
    )
    tqdm = mute_tqdm_cls()
    output_file = (args.env.output_home / f"{args.env.job_name}-{args.env.time_stamp}.jsonl")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        with MongoDBTable(args.table) as out_table, output_file.open("w") as out_file:
            out_table.drop()
            num_input, inputs = args.data.total, args.data.items
            if args.data.limit > 0:
                num_input, inputs = min(args.data.total, args.data.limit), islice(inputs, args.data.limit)
            num_batch, batches = math.ceil(num_input / args.data.batch), ichunked(inputs, args.data.batch)
            logger.info(f"Check {num_input} IP addresses with {num_batch} batches")
            progress, interval = (tqdm(batches, total=num_batch, unit="batch", pre="*", desc="visiting"),
                                  math.ceil(args.data.prog_interval / args.data.batch))
            with ProcessPoolExecutor(max_workers=args.env.max_workers) as pool:
                def make_jobs() -> Iterable[Future]:
                    for i, x in enumerate(progress):
                        if i > 0 and i % interval == 0:
                            logger.info(progress)
                        yield pool.submit(process_many, batch=x, args=args)
                    logger.info(progress)

                for job in make_jobs():
                    job.result(timeout=args.net.waiting_sec)
                terminate_processes(pool)

            # for i, x in enumerate(progress):
            #     if i > 0 and i % interval == 0:
            #         logger.info(progress)
            #     process_many(batch=x, args=args)
            # logger.info(progress)

            find_opt = {}
            num_row, rows = out_table.count_documents(find_opt), out_table.find(find_opt).sort("_id")
            progress, interval = (tqdm(rows, unit="ea", pre="*", desc="exporting", total=num_row),
                                  args.data.prog_interval * 10)
            for i, row in enumerate(progress):
                if i > 0 and i % interval == 0:
                    logger.info(progress)
                out_file.write(json.dumps(row, ensure_ascii=False) + '\n')
            logger.info(progress)
        logger.info(f"Export {num_row}/{num_input} rows to {output_file}")


if __name__ == "__main__":
    app()
