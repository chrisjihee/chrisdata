import json
import logging
import math

import typer
from elasticsearch.helpers import streaming_bulk

from chrisbase.data import AppTyper, JobTimer, ProjectEnv
from chrisbase.data import IOArguments, InputOption, OutputOption, FileOption, TableOption, IndexOption
from chrisbase.data import Streamer, FileStreamer, MongoStreamer, ElasticStreamer
from chrisbase.io import LoggingFormat
from chrisbase.util import mute_tqdm_cls

logger = logging.getLogger(__name__)


class IndexApp:
    app = AppTyper()

    @classmethod
    def typer(cls) -> typer.Typer:

        @cls.app.command()
        def wikipedia(
                # env
                project: str = typer.Option(default="chrisdata"),
                job_name: str = typer.Option(default="index_wikipedia"),
                output_home: str = typer.Option(default="output-index_wikipedia"),
                logging_file: str = typer.Option(default="logging.out"),
                debugging: bool = typer.Option(default=False),
                # input
                input_start: int = typer.Option(default=0),
                input_limit: int = typer.Option(default=-1),
                input_batch: int = typer.Option(default=10000),
                input_inter: int = typer.Option(default=100000),
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
                logging_home=output_home,
                logging_file=logging_file,
                message_level=logging.DEBUG if debugging else logging.INFO,
                message_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_36,
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
                ),
            )
            args = IOArguments(
                env=env,
                input=input_opt,
                output=output_opt,
            )
            tqdm = mute_tqdm_cls()
            assert args.input.file or args.input.table, "input.file or input.table is required"
            assert args.output.index, "output.index is required"
            logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)

            with (
                JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
                MongoStreamer(args.input.table) as input_table, FileStreamer(args.input.file) as input_file,
                ElasticStreamer(args.output.index) as output_index, FileStreamer(args.output.file) as output_file,
            ):
                writer = Streamer.first_usable(output_index, output_file)
                reader = Streamer.first_usable(input_table, input_file)
                input_items: InputOption.InputItems = args.input.ready_inputs(reader, len(reader))
                logger.info(f"Index from [{reader.opt}] to [{writer.opt}]")
                logger.info(f"- amount: {input_items.num_item}{'' if input_items.has_single_items() else f' * {args.input.batch}'} ({type(input_items).__name__})")
                progress, interval = (
                    tqdm(input_items.items, total=input_items.num_item, unit="batch", pre="*", desc="indexing"),
                    math.ceil(args.input.inter / args.input.batch)
                )
                for i, x in enumerate(progress):
                    if i > 0 and i % interval == 0:
                        logger.info(progress)
                    if input_items.has_batch_items():
                        if isinstance(writer, ElasticStreamer):
                            for ok, action in streaming_bulk(writer.cli, actions=x, chunk_size=args.input.batch, index=writer.opt.name, yield_ok=False):
                                logger.warning(f"ok={ok}, action={action}")
                    else:
                        x["passage_id"] = x.pop("_id")
                        if isinstance(writer, ElasticStreamer):
                            writer.cli.index(index=writer.opt.name, document=x)
                        elif isinstance(writer, FileStreamer):
                            writer.fp.write(json.dumps(x, ensure_ascii=False) + "\n")
                logger.info(progress)
                writer.status()
                logger.info(f"Indexed {len(writer)} items to [{writer.opt}]")

        return cls.app


if __name__ == "__main__":
    main = AppTyper()
    main.add_typer(IndexApp.typer(), name="index")
    main()
