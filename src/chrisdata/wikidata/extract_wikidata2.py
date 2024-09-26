import typer

from chrisbase.data import FileStreamer, MongoStreamer
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat, new_path
from chrisbase.util import mute_tqdm_cls
from chrisdata.wikidata import *

logger = logging.getLogger(__name__)


@app.command()
def extract(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="extract_wikidata"),
        logging_home: str = typer.Option(default="output/wikidata/extract"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=True),  # TODO: Replace with False
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),  # TODO: Replace with -1
        input_batch: int = typer.Option(default=1000),
        input_inter: int = typer.Option(default=5000),
        input_total: int = typer.Option(default=105485440),  # https://www.wikidata.org/wiki/Wikidata:Statistics  # TODO: Replace with (actual count)
        input_table_home: str = typer.Option(default="localhost:6382/Wikidata"),
        input_table_name: str = typer.Option(default="wikidata-20230911-all-parse-ko-en"),
        # output
        output_file_home: str = typer.Option(default="output/wikidata"),
        output_file_name: str = typer.Option(default="wikidata-20230911-all-extract.jsonl"),
        output_file_mode: str = typer.Option(default="w"),
        output_table_home: str = typer.Option(default="localhost:6382/Wikidata"),
        output_table_name: str = typer.Option(default="wikidata-20230911-all-extract"),
        output_table_reset: bool = typer.Option(default=True),
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
    input_opt = InputOption(
        start=input_start,
        limit=input_limit if not debugging else 1000,
        batch=input_batch if not debugging else 10,
        inter=input_inter if not debugging else 1,
        total=input_total,
        table=TableOption(
            home=input_table_home,
            name=input_table_name,
            required=True,
        )
    )
    output_opt = OutputOption(
        file=FileOption(
            home=output_file_home,
            name=new_path(output_file_name, post=env.time_stamp),
            mode=output_file_mode,
            required=True,
        ),
        table=TableOption(
            home=output_table_home,
            name=output_table_name,
            reset=output_table_reset,
            required=True,
        )
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.table, "input.table is required"
    assert args.output.file, "output.file is required"
    assert args.output.table, "output.table is required"

    with(
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoStreamer(args.input.table) as input_table,
        FileStreamer(args.output.file) as output_file,
        MongoStreamer(args.output.table) as output_table,
    ):
        # extract time-sensitive triples
        input_data = args.input.ready_inputs(input_table, total=len(input_table))
        logger.info(f"Extract from [{input_table.opt}] to [{output_table.opt}]")
        logger.info(f"- [input] total={args.input.total} | start={args.input.start} | limit={args.input.limit}"
                    f" | {type(input_data).__name__}={input_data.num_item}{f'x{args.input.batch}ea' if input_data.has_batch_items() else ''}")
        logger.info(f"- [output] table.reset={args.output.table.reset} | table.timeout={args.output.table.timeout}")
        with tqdm(total=input_data.num_item, unit="item", pre="=>", desc="extracting", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
            for batch in input_data.items:
                for x in batch:
                    logger.info(f"id={x['_id']}, label={x['label1'] or x['label2']}")
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
