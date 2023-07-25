import logging
import os

import typer

from chrisbase.data import AppTyper, ProjectEnv, CommonArguments, JobTimer
from chrisbase.io import LoggingFormat
from chrisbase.net import check_ip_addrs
from chrisbase.util import MongoDB

logger = logging.getLogger(__name__)
app = AppTyper()


@app.command()
def check(
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        output_home: str = typer.Option(default="output"),
        max_workers: int = typer.Option(default=os.cpu_count()),
):
    args = CommonArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else "check_ip_addrs",
            output_home=output_home,
            debugging=debugging,
            msg_level=logging.DEBUG if debugging else logging.INFO,
            msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
            max_workers=1 if debugging else max(max_workers, 1),
        )
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        with MongoDB(db_name=args.env.project, tab_name=args.env.job_name) as mongo:
            mongo.table.drop()
            check_ip_addrs(args=args, mongo=mongo)
            mongo.output_table(to=args.env.output_home / f"{args.env.job_name}.jsonl", include_id=True)


if __name__ == "__main__":
    app()
