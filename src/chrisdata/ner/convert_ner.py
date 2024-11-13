import json

import typer

from chrisbase.data import ProjectEnv, InputOption, FileOption, OutputOption, IOArguments, JobTimer, FileStreamer
from chrisbase.io import LoggingFormat, new_path
from chrisbase.util import mute_tqdm_cls
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
        # input
        input_file_path: str = typer.Option(default="input/GNER/zero-shot-test.jsonl"),
        # output
        output_file_path: str = typer.Option(default="output/GNER/zero-shot-test-conv.jsonl"),
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
    input_opt = InputOption(
        inter=1,
        limit=1,
        file=FileOption.from_path(
            path=input_file_path,
            required=True,
        ),
    )
    output_opt = OutputOption(
        file=FileOption.from_path(
            path=output_file_path,
            name=new_path(output_file_path, post=env.time_stamp).name,
            mode="w",
            required=True,
        ),
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.file, "input.file is required"
    assert args.output.file, "output.file is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(args.input.file) as input_file,
        FileStreamer(args.output.file) as output_file,
    ):
        input_data = args.input.ready_inputs(input_file, total=len(input_file))
        for x in [json.loads(a) for a in input_data.items]:
            logger.info(json.dumps(x, indent=2))
            logger.info(x['dataset'])
            logger.info(x['split'])
            logger.info(x['label_list'])
            logger.info(x['instance']['id'])
            logger.info(x['instance']['words'])
            logger.info(x['instance']['labels'])
            logger.info(x['instance']['instruction_inputs'])
            logger.info(x['instance']['prompt_labels'])
