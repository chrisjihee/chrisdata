import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
import typer
from elasticsearch.helpers import streaming_bulk

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, CommonArguments
from chrisbase.data import DataOption, FileOption, TableOption, IndexOption
from chrisbase.data import LineFileWrapper, MongoDBWrapper, ElasticSearchWrapper
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class ProgramArguments(CommonArguments):
    data: DataOption = field()

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


def process_many(batch: Iterable[dict], wrapper: ElasticSearchWrapper, batch_size: int):
    for ok, action in streaming_bulk(client=wrapper.cli,
                                     index=wrapper.opt.name,
                                     actions=batch,
                                     chunk_size=batch_size,
                                     yield_ok=False):
        logger.warning(f"ok={ok}, action={action}")


@app.command()
def index(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="index_wikipedia"),
        output_home: str = typer.Option(default="output-index_wikipedia"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # data
        data_start: int = typer.Option(default=0),
        data_limit: int = typer.Option(default=-1),
        # data_limit: int = typer.Option(default=100),
        data_batch: int = typer.Option(default=10000),
        data_inter: int = typer.Option(default=100000),
        data_total: int = typer.Option(default=2013506),
        file_home: str = typer.Option(default="input/wikimedia"),
        file_name: str = typer.Option(default="wikipedia-20230920-parse-kowiki.jsonl"),
        table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        table_name: str = typer.Option(default="wikipedia-20230920-parse-kowiki"),
        index_home: str = typer.Option(default="localhost:9810"),
        index_name: str = typer.Option(default="wikipedia-20230920-index-kowiki"),
        index_user: str = typer.Option(default="elastic"),
        index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
        index_reset: bool = typer.Option(default=True),
        index_create: str = typer.Option(default="input/wikimedia/wikipedia-index_create_args.json"),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        output_home=output_home,
        logging_file=logging_file,
        msg_level=logging.DEBUG if debugging else logging.INFO,
        msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_36,
    )
    data_opt = DataOption(
        start=data_start,
        limit=data_limit,
        batch=data_batch,
        inter=data_inter,
        total=data_total,
        file=FileOption(
            home=file_home,
            name=file_name,
        ) if file_home and file_name else None,
        table=TableOption(
            home=table_home,
            name=table_name,
        ) if table_home and table_name else None,
        index=IndexOption(
            home=index_home,
            user=index_user,
            pswd=index_pswd,
            name=index_name,
            reset=index_reset,
            create=index_create,
        ) if index_home and index_name else None,
    )
    args = ProgramArguments(
        env=env,
        data=data_opt,
    )
    tqdm = mute_tqdm_cls()
    logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
    assert args.data.file or args.data.table, "data.file or data.table is required"
    assert args.data.index, "data.index is required"

    with (
        JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        ElasticSearchWrapper(args.data.index) as data_index,
        MongoDBWrapper(args.data.table) as data_table,
        LineFileWrapper(args.data.file) as data_file,
    ):
        # index parsed data
        if data_table and data_table.usable():
            data_source = args.data.table
            batches, num_batch, num_input = args.data.input_batches(data_table, args.data.total)
        elif data_file and data_file.usable():
            data_source = args.data.file
            batches, num_batch, num_input = args.data.input_batches(data_file, args.data.total)
        else:
            assert False, "No data source"
        logger.info(f"Index from [{data_source}] to [{args.data.index}]")
        logger.info(f"- amount: inputs={num_input}, batches={num_batch}")
        progress, interval = (
            tqdm(batches, total=num_batch, unit="batch", pre="*", desc="indexing"),
            math.ceil(args.data.inter / args.data.batch)
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            process_many(batch=x, wrapper=data_index, batch_size=args.data.batch)
        logger.info(progress)
        data_index.refresh(verbose=True)
        logger.info(f"Indexed {len(data_index)} documents to [{args.data.index}]")


if __name__ == "__main__":
    app()
