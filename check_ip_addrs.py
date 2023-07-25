import logging
import os

import typer
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.typings import _DocumentType

from chrisbase.data import AppTyper, ProjectEnv, CommonArguments, JobTimer
from chrisbase.io import LoggingFormat
from chrisbase.net import check_ip_addrs

logger = logging.getLogger(__name__)
app = AppTyper()


class ResultDB:
    def __init__(self, db_name, tab_name, host="localhost", port=27017):
        self.db_name = db_name
        self.tab_name = tab_name
        self.host = host
        self.port = port
        self.mongo: MongoClient[_DocumentType] | None = None
        self.table: Collection | None = None

    def __enter__(self):
        pass

    def __exit__(self, *exc_info):
        pass


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
            job_name=job_name if job_name else "check-ip-addrs",
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
