import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
import typer
from elasticsearch.helpers import streaming_bulk

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, CommonArguments
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption, IndexOption
from chrisbase.data import LineFileWrapper, MongoDBWrapper, ElasticSearchWrapper
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class IndexArguments(CommonArguments):
    input: InputOption = field()
    output: OutputOption = field()

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.input, data_prefix="input", data_exclude=["file", "table", "index"]),
            to_dataframe(columns=columns, raw=self.input.file, data_prefix="input.file") if self.input.file else None,
            to_dataframe(columns=columns, raw=self.input.table, data_prefix="input.table") if self.input.table else None,
            to_dataframe(columns=columns, raw=self.output.index, data_prefix="output.index") if self.output.index else None,
        ]).reset_index(drop=True)


def index_many(batch: Iterable[dict], wrapper: ElasticSearchWrapper, batch_size: int):
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
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=10000),
        input_inter: int = typer.Option(default=100000),
        input_total: int = typer.Option(default=2013506),
        input_file_home: str = typer.Option(default="input/wikimedia"),
        input_file_name: str = typer.Option(default="wikipedia-20230920-parse-kowiki.jsonl"),
        input_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        input_table_name: str = typer.Option(default="wikipedia-20230920-parse-kowiki"),
        output_index_home: str = typer.Option(default="localhost:9810"),
        output_index_name: str = typer.Option(default="wikipedia-20230920-index-kowiki"),
        output_index_user: str = typer.Option(default="elastic"),
        output_index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
        output_index_reset: bool = typer.Option(default=True),
        output_index_create: str = typer.Option(default="input/wikimedia/wikipedia-index_create_args.json"),
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
    input_opt = InputOption(
        start=input_start,
        limit=input_limit,
        batch=input_batch,
        inter=input_inter,
        total=input_total,
        file=FileOption(
            home=input_file_home,
            name=input_file_name,
        ) if input_file_home and input_file_name else None,
        table=TableOption(
            home=input_table_home,
            name=input_table_name,
        ) if input_table_home and input_table_name else None,
    )
    output_opt = OutputOption(
        index=IndexOption(
            home=output_index_home,
            user=output_index_user,
            pswd=output_index_pswd,
            name=output_index_name,
            reset=output_index_reset,
            create=output_index_create,
        ) if output_index_home and output_index_name else None,
    )
    args = IndexArguments(
        env=env,
        input=input_opt,
        output=output_opt,
    )
    tqdm = mute_tqdm_cls()
    logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
    assert args.input.file or args.input.table, "input.file or input.table is required"
    assert args.output.index, "output.index is required"

    with (
        JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoDBWrapper(args.input.table) as input_table, LineFileWrapper(args.input.file) as input_file,
        ElasticSearchWrapper(args.output.index) as output_index,
    ):
        # index parsed data
        inputs = args.input.select_inputs(input_table, input_file)
        logger.info(f"Index from [{inputs.wrapper.opt}] to [{args.output.index}]")
        logger.info(f"- amount: inputs={inputs.num_input}, batches={inputs.num_batch}")
        progress, interval = (
            tqdm(inputs.batches, total=inputs.num_batch, unit="batch", pre="*", desc="indexing"),
            math.ceil(args.input.inter / args.input.batch)
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            index_many(batch=x, wrapper=output_index, batch_size=args.input.batch)
        logger.info(progress)
        output_index.refresh(verbose=True)
        logger.info(f"Indexed {len(output_index)} documents to [{args.output.index}]")


if __name__ == "__main__":
    app()
