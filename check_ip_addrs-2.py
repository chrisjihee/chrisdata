import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import islice

import httpx
import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments, TableOption, MongoStreamer
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls, wait_future_jobs

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class DataOption(OptionData):
    items: enumerate = field()
    total: int = field(default=0)
    limit: int = field(default=-1)
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
    _id: int
    query: str
    status: int | None = None
    elapsed: float | None = None
    size: float | None = None
    text: str | None = None


def process_one(i: int, x: str, args: ProgramArguments):
    if args.net and args.net.calling_sec > 0:
        time.sleep(args.net.calling_sec)
    with httpx.Client(
            transport=httpx.HTTPTransport(local_address=x),
            timeout=httpx.Timeout(timeout=120.0)
    ) as cli:
        response = cli.get("https://api64.ipify.org?format=json")
        result = ProcessResult(
            _id=i,
            query=x,
            status=response.status_code,
            elapsed=round(response.elapsed.total_seconds(), 3),
            size=round(response.num_bytes_downloaded / 1024, 6),
            text=response.text,
        )
        with MongoStreamer(args.table) as db:
            db.table.insert_one(result.to_dict())


@app.command()
def check(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="check_ip_addrs"),
        output_home: str = typer.Option(default="output-check_ip_addrs"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=10),
        debugging: bool = typer.Option(default=False),
        # net
        calling_sec: float = typer.Option(default=0.001),
        waiting_sec: float = typer.Option(default=300.0),
        # data
        input_limit: int = typer.Option(default=-1),
        prog_interval: int = typer.Option(default=10),
        # table
        table_home: str = typer.Option(default="localhost:6382/check"),
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
    assert env.num_ip_addrs > 0, f"env.num_ip_addrs={env.num_ip_addrs}"
    args = ProgramArguments(
        env=env,
        net=NetOption(
            calling_sec=calling_sec,
            waiting_sec=waiting_sec,
        ),
        data=DataOption(
            items=enumerate(env.ip_addrs, start=1),
            total=env.num_ip_addrs,
            limit=input_limit,
            prog_interval=prog_interval if prog_interval > 0 else env.max_workers,
        ),
        table=TableOption(
            home=table_home,
            name=env.job_name,
        ),
    )
    tqdm = mute_tqdm_cls()
    output_file = (args.env.output_home / f"{args.env.job_name}-{args.env.time_stamp}.jsonl")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    with JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        with MongoStreamer(args.table) as out_table, output_file.open("w") as out_file:
            out_table.reset()
            input_list = islice(args.data.items, args.data.limit) if args.data.limit > 0 else args.data.items
            input_size = min(args.data.total, args.data.limit) if args.data.limit > 0 else args.data.total
            logger.info(f"Use {args.env.max_workers} workers to check {input_size} IP addresses")
            with ProcessPoolExecutor(max_workers=args.env.max_workers) as pool:
                jobs = [(i, pool.submit(process_one, i=i, x=x, args=args)) for i, x in input_list]
                prog_bar = tqdm(jobs, unit="ea", pre="*", desc="visiting")
                wait_future_jobs(prog_bar, timeout=args.net.waiting_sec, interval=args.data.prog_interval, pool=pool)

            row_filter = {}
            num_row, rows = out_table.table.count_documents(row_filter), out_table.table.find(row_filter).sort("_id")
            prog_bar = tqdm(rows, unit="ea", pre="*", desc="exporting", total=num_row)
            for i, row in enumerate(prog_bar, start=1):
                out_file.write(json.dumps(row, ensure_ascii=False) + '\n')
                if i % (args.data.prog_interval * 10) == 0:
                    logger.info(prog_bar)
            logger.info(prog_bar)
            logger.info(f"Export {num_row}/{input_size} rows to {output_file}")


if __name__ == "__main__":
    app()
