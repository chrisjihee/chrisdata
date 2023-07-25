import logging

import typer

from chrisbase.data import AppTyper, ProjectEnv, CommonArguments, JobTimer
from chrisbase.io import LoggingFormat
from chrisbase.net import ips, check_ip_addrs

logger = logging.getLogger(__name__)
app = AppTyper()


@app.command()
def check(
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        output_home: str = typer.Option(default="output"),
):
    args = CommonArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else f"IP Check",
            output_home=output_home,
            debugging=debugging,
            msg_level=logging.DEBUG if debugging else logging.INFO,
            msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
        )
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        args.info_arguments()
        logger.info(f"Check {len(ips)} IP addresses")
        check_ip_addrs()


if __name__ == "__main__":
    app()
