import json
import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
import typer
from elasticsearch.helpers import streaming_bulk

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, CommonArguments, GenericRewriter
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption, IndexOption
from chrisbase.data import FileRewriter, MongoRewriter, ElasticRewriter
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls
from parse_wikipedia import PassageUnit

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


#
#
# def index_many(batch: Iterable[dict], wrapper: ElasticRewriter, batch_size: int):
#     for ok, action in streaming_bulk(client=wrapper.cli,
#                                      index=wrapper.opt.name,
#                                      actions=batch,
#                                      chunk_size=batch_size,
#                                      yield_ok=False):
#         logger.warning(f"ok={ok}, action={action}")


@app.command()
def index(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="index_wikipedia"),
        output_home: str = typer.Option(default="output-index_wikipedia"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=500),
        input_batch: int = typer.Option(default=1),
        input_inter: int = typer.Option(default=100),
        # input_limit: int = typer.Option(default=-1),
        # input_batch: int = typer.Option(default=10000),
        # input_inter: int = typer.Option(default=100000),
        input_file_home: str = typer.Option(default="input/wikimedia"),
        input_file_name: str = typer.Option(default="wikipedia-20230920-parse-kowiki.jsonl"),  # TODO: 문장 단위로 색인해보거나, 형태소분석기를 적용한 후 색인해보기
        input_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        input_table_name: str = typer.Option(default="wikipedia-20230920-parse-kowiki"),
        # output
        output_file_home: str = typer.Option(default="output-index_wikipedia"),
        output_file_name: str = typer.Option(default="wikipedia-20230920-index-kowiki.jsonl"),
        output_file_reset: bool = typer.Option(default=False),
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
        file=FileOption(
            home=input_file_home,
            name=input_file_name,
        ),
        table=TableOption(
            home=input_table_home,
            name=input_table_name,
        ),
    )
    output_opt = OutputOption(
        file=FileOption(
            home=output_file_home,
            name=output_file_name,
            reset=output_file_reset,
            mode="a",
        ),
        index=IndexOption(
            home=output_index_home,
            user=output_index_user,
            pswd=output_index_pswd,
            name=output_index_name,
            reset=output_index_reset,
            create=output_index_create,
            strict=False,
        ),
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
        MongoRewriter(args.input.table) as input_table, FileRewriter(args.input.file) as input_file,
        ElasticRewriter(args.output.index) as output_index, FileRewriter(args.output.file) as output_file,
    ):
        # index parsed data
        reader = GenericRewriter.first_usable(input_table, input_file)
        writer = GenericRewriter.first_usable(output_index, output_file)
        logger.info(f"len(input_table)={len(input_table)}")
        logger.info(f"len(output_index)={len(output_index)}")
        logger.info(f"Index from [{reader.opt}] to [{writer.opt}]")
        input_items: InputOption.InputItems = args.input.ready_inputs(reader, len(reader))
        logger.info(f"- amount: {input_items.total}{'' if isinstance(input_items, InputOption.SingleItems) else f' * {args.input.batch}'} ({type(input_items).__name__})")
        progress, interval = (
            tqdm(input_items.items, total=input_items.total, unit="batch", pre="*", desc="indexing"),
            math.ceil(args.input.inter / args.input.batch)
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            # logger.info(f"i={i}")
            # for a in x:
            #     logger.info(f"- a={a}")
            # print(type(writer))
            # print(type(x))
            x["passage_id"] = x.pop("_id")
            if isinstance(writer, ElasticRewriter):
                writer.cli.index(index=writer.opt.name, document=x)
            elif isinstance(writer, FileRewriter):
                writer.fp.write(json.dumps(x, ensure_ascii=False) + "\n")
            # index_many(batch=x, wrapper=output_index, batch_size=args.input.batch)
        logger.info(progress)
        output_index.refresh()
        logger.info(f"Indexed {len(writer)} items to [{writer.opt}]")
        writer.status(only_opt=False)


if __name__ == "__main__":
    app()
#
#
# def index_many(batch: Iterable[dict], wrapper: ElasticRewriter, batch_size: int):
#     for ok, action in streaming_bulk(client=wrapper.cli,
#                                      index=wrapper.opt.name,
#                                      actions=batch,
#                                      chunk_size=batch_size,
#                                      yield_ok=False):
#         logger.warning(f"ok={ok}, action={action}")
