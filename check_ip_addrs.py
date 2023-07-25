import logging
import os

import typer

from chrisbase.data import AppTyper, ProjectEnv, CommonArguments, JobTimer
from chrisbase.io import LoggingFormat
from chrisbase.net import check_ip_addrs

logger = logging.getLogger(__name__)
app = AppTyper()


@app.command()
def check(
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        output_home: str = typer.Option(default="output"),
        num_workers: int = typer.Option(default=os.cpu_count()),
):
    args = CommonArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else f"IPCheck",
            output_home=output_home,
            debugging=debugging,
            msg_level=logging.DEBUG if debugging else logging.INFO,
            msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
        )
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        check_ip_addrs(args=args, num_workers=num_workers)


if __name__ == "__main__":
    app()
