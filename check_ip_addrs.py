import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, field
from typing import List

import httpx
import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments
from chrisbase.io import LoggingFormat
from chrisbase.util import MongoDB, to_dataframe, time_tqdm_cls, mute_tqdm_cls, wait_future_jobs

logger = logging.getLogger(__name__)
app = AppTyper()
mongos: List[MongoDB] = []


@dataclass
class NetOption(OptionData):
    calling_sec: float = field(default=1.0)
    waiting_sec: float = field(default=30.0),


@dataclass
class ProgramArguments(CommonArguments):
    net: NetOption | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.net, data_prefix="net"),
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
        ]).reset_index(drop=True)


@dataclass
class ProcessResult(DataClassJsonMixin):
    _id: int
    query: str
    status: int | None = None
    elapsed: float | None = None
    size: float | None = None
    text: str | None = None


def process_query(i: int, x: str, s: float | None = None, log: bool = True):
    if s and s > 0:
        time.sleep(s)
    with httpx.Client(transport=httpx.HTTPTransport(local_address=x), timeout=10.0) as cli:
        response = cli.get("https://api64.ipify.org?format=json")
        result = ProcessResult(
            _id=i,
            query=x,
            status=response.status_code,
            elapsed=round(response.elapsed.total_seconds(), 3),
            size=round(response.num_bytes_downloaded / 1024, 6),
            text=response.text,
        )
        for mongo in mongos:
            mongo.table.insert_one(result.to_dict())
        if log:
            logger.info("  * " + ' ----> '.join([
                f"[{result.query:<15s}]",
                f"[{result.status}]",
                f"[{result.elapsed * 1000:7,.0f}ms]",
                f"[{result.size:7,.2f}KB]",
                f"{result.text}",
            ]))


@app.command()
def check(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="check_ip_addrs"),
        debugging: bool = typer.Option(default=False),
        max_workers: int = typer.Option(default=os.cpu_count()),
        output_home: str = typer.Option(default="output-check_ip_addrs"),
        # net
        calling_sec: float = typer.Option(default=1.0),
        waiting_sec: float = typer.Option(default=30.0),
        # etc
        use_tqdm: bool = typer.Option(default=False),
):
    args = ProgramArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name,
            debugging=debugging,
            output_home=output_home,
            msg_level=logging.DEBUG if debugging else logging.INFO,
            msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
            max_workers=1 if debugging else max(max_workers, 1),
        ),
        net=NetOption(
            calling_sec=calling_sec,
            waiting_sec=waiting_sec,
        ),
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        with MongoDB(db_name=args.env.project, tab_name=args.env.job_name, clear_table=True, pool=mongos) as mongo:
            logger.info(f"Use {args.env.max_workers} workers to check {args.env.num_ip_addrs} IP addresses")
            job_tqdm = time_tqdm_cls(bar_size=100, desc_size=9) if use_tqdm else mute_tqdm_cls()
            if args.env.max_workers < 2:
                for i, x in enumerate(job_tqdm(args.env.ip_addrs, pre="┇", desc="visiting", unit="job")):
                    process_query(i=i + 1, x=x, log=not use_tqdm)
            else:
                with ProcessPoolExecutor(max_workers=args.env.max_workers) as pool:
                    jobs: List[Future] = []
                    for i, x in enumerate(args.env.ip_addrs):
                        jobs.append(pool.submit(process_query, i=i + 1, x=x, s=args.net.calling_sec, log=not use_tqdm))
                    wait_future_jobs(job_tqdm(jobs, pre="┇", desc="visiting", unit="job"), timeout=args.net.waiting_sec, pool=pool)
            logger.info(f"Success: {mongo.num_documents}/{args.env.num_ip_addrs}")
            mongo.output_table(to=args.env.output_home / f"{args.env.job_name}.jsonl")


if __name__ == "__main__":
    app()
