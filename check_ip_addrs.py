import logging
import os

import typer

from chrisbase.data import AppTyper, ProjectEnv, CommonArguments, JobTimer
from chrisbase.io import LoggingFormat
from chrisbase.net import ips, num_ip_addrs, check_ip_addr
from chrisbase.util import MongoDB

logger = logging.getLogger(__name__)
app = AppTyper()


@app.command()
def check(
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="check_ip_addrs"),
        debugging: bool = typer.Option(default=False),
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
        with MongoDB(db_name=args.env.project, tab_name=args.env.job_name, clear_table=True) as mongo:
            logger.info(f"Use {args.env.max_workers} workers to check {num_ip_addrs()} IP addresses")
            if args.env.max_workers < 2:
                for i, ip in enumerate(ips):
                    res = check_ip_addr(ip=ip, _id=i + 1)
                    mongo.table.insert_one(res)
            else:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                pool = ProcessPoolExecutor(max_workers=args.env.max_workers)
                jobs = [pool.submit(check_ip_addr, ip=ip, _id=i + 1) for i, ip in enumerate(ips)]
                for job in as_completed(jobs):
                    mongo.table.insert_one(job.result())
            mongo.output_table(to=args.env.output_home / f"{args.env.job_name}.jsonl", include_id=True)


if __name__ == "__main__":
    app()
