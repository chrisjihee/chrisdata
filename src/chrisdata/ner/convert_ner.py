import typer

from chrisbase.data import ProjectEnv
from chrisbase.io import LoggingFormat
from . import *

logger = logging.getLogger(__name__)


@app.command()
def convert(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="convert_wikidata"),
        logging_home: str = typer.Option(default="output/wikidata/convert"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        logging_home=logging_home,
        logging_file=logging_file,
        message_level=logging.INFO,
        message_format=LoggingFormat.CHECK_00,  # if not debugging else LoggingFormat.DEBUG_36,
        max_workers=1 if debugging else max(max_workers, 1),
    )
    logger.info("ENV: %s", env)
