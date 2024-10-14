import json

import pandas as pd
import typer

from chrisbase.data import FileStreamer
from chrisbase.data import InputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv, CommonArguments
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls
from . import *

logger = logging.getLogger(__name__)


@dataclass
class RestoreArguments(CommonArguments):
    data: InputOption = field()

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data", data_exclude=["file", "table", "index"]),
            to_dataframe(columns=columns, raw=self.data.file, data_prefix="data.file") if self.data.file else None,
            to_dataframe(columns=columns, raw=self.data.table, data_prefix="data.table") if self.data.table else None,
            to_dataframe(columns=columns, raw=self.data.index, data_prefix="data.index") if self.data.index else None,
        ]).reset_index(drop=True)


@app.command()
def restore(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="restore_wikidata"),
        output_home: str = typer.Option(default="output-restore_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # data
        data_start: int = typer.Option(default=0),
        data_limit: int = typer.Option(default=-1),
        data_batch: int = typer.Option(default=10000),
        data_inter: int = typer.Option(default=100000),
        data_total: int = typer.Option(default=1018174),
        file_home: str = typer.Option(default="input/wikimedia"),
        file_name: str = typer.Option(default="wikidata-20230920-parse-kowiki.jsonl"),
        table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        table_name: str = typer.Option(default="wikidata-20230920-parse-kowiki"),
        table_reset: bool = typer.Option(default=True),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        logging_home=output_home,
        logging_file=logging_file,
        message_level=logging.INFO,  # if not debugging else logging.DEBUG,
        message_format=LoggingFormat.CHECK_24,  # if not debugging else LoggingFormat.DEBUG_24,
    )
    data_opt = InputOption(
        start=data_start,
        limit=data_limit,
        batch=data_batch,
        inter=data_inter,
        file=FileOption(
            home=file_home,
            name=file_name,
        ) if file_home and file_name else None,
        table=TableOption(
            home=table_home,
            name=table_name,
            reset=table_reset,
        ) if table_home and table_name else None,
    )
    args = RestoreArguments(
        env=env,
        data=data_opt,
    )
    tqdm = mute_tqdm_cls()
    save_file = (args.env.logging_home / f"{table_name}-{args.env.time_stamp}.jsonl")
    assert args.data.file, "data.file is required"
    assert args.data.table, "data.table is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoStreamer(args.data.table) as data_table,
        FileStreamer(args.data.file) as data_file,
        save_file.open("w") as writer,
    ):
        # restore parsed data
        inputs = args.data.ready_inputs(data_file, data_total)
        logger.info(f"Restore from [{args.data.file}] to [{args.data.table}]")
        logger.info(f"- amount: inputs={inputs.num_input}, batches={inputs.total}")
        progress, interval = (
            tqdm(inputs.batches, total=inputs.total, unit="batch", pre="*", desc="restoring"),
            math.ceil(args.data.inter / args.data.batch),
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            data_table.table.insert_many(x)
        logger.info(progress)

        # save restored data
        progress, interval = (
            tqdm(data_table, total=len(data_table), unit="row", pre="*", desc="saving"),
            args.data.inter,
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            writer.write(json.dumps(x, ensure_ascii=False) + '\n')
        logger.info(progress)
        logger.info(f"Saved {len(data_table)} rows to [{save_file}]")
