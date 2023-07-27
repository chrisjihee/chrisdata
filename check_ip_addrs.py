import logging
import os
from concurrent.futures import ProcessPoolExecutor, Future
from typing import List

import httpx
import typer

from chrisbase.data import AppTyper, ProjectEnv, CommonArguments, JobTimer
from chrisbase.io import LoggingFormat
from chrisbase.net import ips, num_ip_addrs
from chrisbase.proc import all_future_results
from chrisbase.util import MongoDB

logger = logging.getLogger(__name__)
app = AppTyper()
savers: List[MongoDB] = []


def check_local_address(i: int, x: str, log: bool = True) -> int:
    with httpx.Client(transport=httpx.HTTPTransport(local_address=x)) as cli:
        response = cli.get("https://api64.ipify.org?format=json", timeout=10.0)
        result = {
            '_id': i,
            'local_address': x,
            'status': response.status_code,
            'elapsed': round(response.elapsed.total_seconds(), 3),
            'size': round(response.num_bytes_downloaded / 1024, 6),
            'text': response.text,
        }
        if log:
            logger.info("  * " + ' ----> '.join([
                f"[{result['local_address']:<15s}]",
                f"[{result['status']}]",
                f"[{result['elapsed'] * 1000:7,.0f}ms]",
                f"[{result['size']:7,.2f}KB]",
                f"{result['text']}",
            ]))
        if response.status_code == 200:
            for saver in savers:
                saver.table.insert_one(result)
        return 1 if response.status_code == 200 else 0


@app.command()
def check(
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="check_ip_addrs"),
        debugging: bool = typer.Option(default=False),
        timeout: float = typer.Option(default=10.0),
        max_workers: int = typer.Option(default=os.cpu_count()),
        output_home: str = typer.Option(default="output-check_ip_addrs"),
):
    args = CommonArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name,
            debugging=debugging,
            output_home=output_home,
            msg_level=logging.DEBUG if debugging else logging.INFO,
            msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
            max_workers=1 if debugging else max(max_workers, 1),
        )
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        with MongoDB(db_name=args.env.project, tab_name=args.env.job_name, clear_table=True, pool=savers) as mongo:
            logger.info(f"Use {args.env.max_workers} workers to check {num_ip_addrs()} IP addresses")
            if args.env.max_workers < 2:
                num_success = sum(check_local_address(i=i + 1, x=x, log=True) for i, x in enumerate(ips))
            else:
                pool: ProcessPoolExecutor = ProcessPoolExecutor(max_workers=args.env.max_workers)
                jobs: List[Future] = [pool.submit(check_local_address, i=i + 1, x=x, log=True) for i, x in enumerate(ips)]
                num_success = sum(all_future_results(pool, jobs, default=0, timeout=timeout, use_tqdm=False))
            logger.info(f"Success: {num_success}/{len(ips)}")
            mongo.output_table(to=args.env.output_home / f"{args.env.job_name}.jsonl")


if __name__ == "__main__":
    app()
