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
from parse_wikidata import WikidataUnit

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class SearchArguments(CommonArguments):
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
            to_dataframe(columns=columns, raw=self.input.index, data_prefix="input.index") if self.input.index else None,
            to_dataframe(columns=columns, raw=self.output.file, data_prefix="output.file") if self.output.file else None,
            to_dataframe(columns=columns, raw=self.output.table, data_prefix="output.table") if self.output.table else None,
        ]).reset_index(drop=True)


def process_many(batch: Iterable[dict], wrapper: ElasticSearchWrapper, batch_size: int):
    for ok, action in streaming_bulk(client=wrapper.cli,
                                     index=wrapper.opt.name,
                                     actions=batch,
                                     chunk_size=batch_size,
                                     yield_ok=False):
        logger.warning(f"ok={ok}, action={action}")


@app.command()
def search(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="search_wikidata"),
        output_home: str = typer.Option(default="output-search_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # data
        input_start: int = typer.Option(default=0),
        # input_limit: int = typer.Option(default=-1),
        input_limit: int = typer.Option(default=20000),
        input_batch: int = typer.Option(default=10000),
        input_inter: int = typer.Option(default=100000),
        input_total: int = typer.Option(default=1018174),
        input_file_home: str = typer.Option(default="input/wikimedia"),
        input_file_name: str = typer.Option(default="wikidata-20230920-parse-kowiki.jsonl"),
        input_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        input_table_name: str = typer.Option(default="wikidata-20230920-parse-kowiki"),
        input_index_home: str = typer.Option(default="localhost:9810"),
        input_index_name: str = typer.Option(default="wikipedia-20230920-index-kowiki"),
        input_index_user: str = typer.Option(default="elastic"),
        input_index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
        output_file_home: str = typer.Option(default="input/wikimedia"),
        output_file_name: str = typer.Option(default="wikidata-20230920-search-kowiki.jsonl"),
        output_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        output_table_name: str = typer.Option(default="wikidata-20230920-search-kowiki"),
        output_table_reset: bool = typer.Option(default=True),
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
        index=IndexOption(
            home=input_index_home,
            user=input_index_user,
            pswd=input_index_pswd,
            name=input_index_name,
        ) if input_index_home and input_index_name else None,
    )
    output_opt = OutputOption(
        file=FileOption(
            home=output_file_home,
            name=output_file_name,
        ) if output_file_home and output_file_name else None,
        table=TableOption(
            home=output_table_home,
            name=output_table_name,
            reset=output_table_reset,
        ) if output_table_home and output_table_name else None,
    )
    args = SearchArguments(
        env=env,
        input=input_opt,
        output=output_opt,
    )
    tqdm = mute_tqdm_cls()
    logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
    assert args.input.index, "input.index is required"
    assert args.input.file or args.input.table, "input.file or input.table is required"
    assert args.output.file or args.output.table, "output.file or output.table is required"

    with (
        JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoDBWrapper(args.output.table) as output_table, LineFileWrapper(args.output.file) as output_file,
        MongoDBWrapper(args.input.table) as input_table, LineFileWrapper(args.input.file) as input_file,
        ElasticSearchWrapper(args.input.index) as input_index,
    ):
        # search parsed data
        inputs = args.input.select_inputs(input_table, input_file)
        outputs = args.output.select_outputs(output_table, output_file)
        logger.info(f"Search [{inputs.wrapper.opt}] from [{args.input.index}] to [{outputs.wrapper.opt}]")
        logger.info(f"- amount: inputs={inputs.num_input}, batches={inputs.num_batch}")
        progress, interval = (
            tqdm(inputs.batches, total=inputs.num_batch, unit="batch", pre="*", desc="searching"),
            math.ceil(args.input.inter / args.input.batch)
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            # process_many(batch=x, wrapper=data_index, batch_size=args.data.batch)
            for a in x:
                a = WikidataUnit.from_dict(a)
                print("a", type(a), a)
        logger.info(progress)
        # logger.info(f"Indexed {len(data_index)} documents to [{args.data.index}]")


if __name__ == "__main__":
    app()
