import json
from typing import Iterable

import bson.json_util
import typer
from elasticsearch.helpers import streaming_bulk

from chrisbase.data import InputOption, OutputOption, TableOption, IndexOption
from chrisbase.data import JobTimer, ProjectEnv, FileOption, FileStreamer
from chrisbase.data import Streamer, ElasticStreamer
from chrisbase.io import LoggingFormat
from chrisbase.util import mute_tqdm_cls
from . import *

logger = logging.getLogger(__name__)
app = AppTyper()


class ExtractApp:
    app = AppTyper()

    @classmethod
    def typer(cls) -> typer.Typer:

        def extract_one(x: dict, reader: Streamer):
            single1 = SingleTriple.from_dict(x)
            if single1.entity1.entity != single1.entity2.entity:
                bridge = single1.entity2
                if isinstance(reader, ElasticStreamer):
                    res = reader.cli.search(index=reader.opt.name, query={
                        "match": {"entity1.entity": bridge.entity}
                    })
                    if res.meta.status == 200 and len(res.body["hits"]["hits"]) > 0:
                        for y in res.body["hits"]["hits"]:
                            y = y["_source"]
                            single2 = SingleTriple.from_dict(y)
                            double = DoubleTriple.from_triples(single1, single2)
                            if double:
                                yield double
                elif isinstance(reader, MongoStreamer):
                    for y in reader.table.find({"entity1.entity": bridge.entity}):
                        single2 = SingleTriple.from_dict(y)
                        double = DoubleTriple.from_triples(single1, single2)
                        if double:
                            yield double

        def extract_many(batch: Iterable[dict], writer: Streamer, reader: Streamer):
            all_units = list()
            for x in batch:
                for r in extract_one(x, reader):
                    all_units.append(r)
            rows = [row.to_dict() for row in all_units if row]
            if len(rows) > 0:
                if isinstance(writer, ElasticStreamer):
                    for ok, action in streaming_bulk(writer.cli, actions=rows, chunk_size=len(rows), index=writer.opt.name, yield_ok=False):
                        logger.warning(f"ok={ok}, action={action}")
                elif isinstance(writer, MongoStreamer):
                    writer.table.insert_many(rows)
                else:
                    raise ValueError(f"Unsupported writer: {type(writer)}")

        @cls.app.command()
        def run(
                # env
                project: str = typer.Option(default="chrisdata"),
                job_name: str = typer.Option(default="extract_wikidata"),
                output_home: str = typer.Option(default="output-extract_wikidata"),
                logging_file: str = typer.Option(default="extract.out"),
                debugging: bool = typer.Option(default=False),
                # input
                input_start: int = typer.Option(default=0),
                input_limit: int = typer.Option(default=-1),
                input_batch: int = typer.Option(default=1000),
                input_inter: int = typer.Option(default=5000),
                input_index_home: str = typer.Option(default="localhost:9810"),
                input_index_name: str = typer.Option(default="wikidata-20230920-search-kowiki"),
                input_index_user: str = typer.Option(default="elastic"),
                input_index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
                input_index_sort: str = typer.Option(default="hits:desc"),
                input_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
                input_table_name: str = typer.Option(default="wikidata-20230920-search-kowiki"),
                # output
                output_index_home: str = typer.Option(default="localhost:9810"),
                output_index_name: str = typer.Option(default="wikidata-20230920-extract-kowiki"),
                output_index_user: str = typer.Option(default="elastic"),
                output_index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
                output_index_reset: bool = typer.Option(default=True),
                output_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
                output_table_name: str = typer.Option(default="wikidata-20230920-extract-kowiki"),
                output_table_reset: bool = typer.Option(default=True),
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
                index=IndexOption(
                    home=input_index_home,
                    user=input_index_user,
                    pswd=input_index_pswd,
                    name=input_index_name,
                    sort=input_index_sort,
                ),
                table=TableOption(
                    home=input_table_home,
                    name=input_table_name,
                ),
            )
            output_opt = OutputOption(
                table=TableOption(
                    home=output_table_home,
                    name=output_table_name,
                    reset=output_table_reset,
                    required=True,
                ),
            )
            args = IOArguments(
                env=env,
                input=input_opt,
                output=output_opt,
            )
            tqdm = mute_tqdm_cls()
            assert args.input.index or args.input.table, "input.index or input.table is required"
            assert args.output.index or args.output.table, "output.index or output.table is required"
            logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)

            with (
                JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
                ElasticStreamer(args.input.index) as input_index, MongoStreamer(args.input.table) as input_table,
                MongoStreamer(args.output.table) as output_table,
            ):
                # extract connected triple pairs
                writer = Streamer.first_usable(output_table)
                reader = Streamer.first_usable(input_index, input_table)
                input_items: InputOption.InputItems = args.input.ready_inputs(reader, len(reader))
                logger.info(f"Run ExtractApp")
                logger.info(f"- from: [{type(reader).__name__}] [{reader.opt}]({len(reader)})")
                logger.info(f"  => amount: {input_items.num_item}{'' if input_items.has_single_items() else f' * {args.input.batch}'} ({type(input_items).__name__})")
                logger.info(f"- into: [{type(writer).__name__}] [{writer.opt}]({len(writer)})")
                progress, interval = (
                    tqdm(input_items.items, total=input_items.num_item, unit="batch", pre="*", desc="extracting"),
                    math.ceil(args.input.inter / args.input.batch),
                )
                for i, batch in enumerate(progress):
                    if i > 0 and i % interval == 0:
                        logger.info(progress)
                    extract_many(batch=batch, writer=writer, reader=reader)
                logger.info(progress)
                if isinstance(writer, MongoStreamer):
                    logger.info(f"Inserted {len(writer)} items to [{writer.opt}]")
                elif isinstance(writer, ElasticStreamer):
                    writer.status()
                    logger.info(f"Indexed {len(writer)} items to [{writer.opt}]")
                # logger.info(f"* Writer({len(writer)}):")
                # for x in writer:
                #     logger.info(f"- x={x}")

        return cls.app


class ExportApp:
    app = AppTyper()

    @classmethod
    def typer(cls) -> typer.Typer:

        @cls.app.command()
        def run(
                # env
                project: str = typer.Option(default="chrisdata"),
                job_name: str = typer.Option(default="extract_wikidata"),
                output_home: str = typer.Option(default="output-extract_wikidata"),
                logging_file: str = typer.Option(default="export.out"),
                debugging: bool = typer.Option(default=False),
                # input
                input_batch: int = typer.Option(default=1),
                input_inter: int = typer.Option(default=5000),
                input_index_home: str = typer.Option(default="localhost:9810"),
                input_index_name: str = typer.Option(default="wikidata-20230920-extract-kowiki"),
                input_index_user: str = typer.Option(default="elastic"),
                input_index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
                input_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
                input_table_name: str = typer.Option(default="wikidata-20230920-extract-kowiki"),
                # output
                output_file_home: str = typer.Option(default="output-extract_wikidata"),
                output_file_name: str = typer.Option(default="wikidata-20230920-extract-kowiki-new.jsonl"),
                output_file_mode: str = typer.Option(default="w"),
                output_file_reset: bool = typer.Option(default=False),
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
                batch=input_batch,
                inter=input_inter,
                index=IndexOption(
                    home=input_index_home,
                    user=input_index_user,
                    pswd=input_index_pswd,
                    name=input_index_name,
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
                    mode=output_file_mode,
                    reset=output_file_reset,
                    required=True,
                ),
            )
            args = IOArguments(
                env=env,
                input=input_opt,
                output=output_opt,
            )
            tqdm = mute_tqdm_cls()
            assert args.input.index or args.input.table, "input.index or input.table is required"
            assert args.output.file, "output.file is required"
            logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)

            with (
                JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
                MongoStreamer(args.input.table) as input_table,
                FileStreamer(args.output.file) as output_file,
            ):
                # export search results
                writer = Streamer.first_usable(output_file)
                reader = Streamer.first_usable(input_table)
                input_items: InputOption.InputItems = args.input.ready_inputs(reader, len(reader))
                logger.info(f"Run ExportApp")
                logger.info(f"- from: [{type(reader).__name__}] [{reader.opt}]({len(reader)})")
                logger.info(f"  => amount: {input_items.num_item}{'' if input_items.has_single_items() else f' * {args.input.batch}'} ({type(input_items).__name__})")
                logger.info(f"- into: [{type(writer).__name__}] [{writer.opt}]({len(writer)})")
                progress, interval = (
                    tqdm(input_items.items, total=input_items.num_item, unit="batch", pre="*", desc="exporting"),
                    math.ceil(args.input.inter / args.input.batch)
                )
                for i, x in enumerate(progress):
                    if i > 0 and i % interval == 0:
                        logger.info(progress)
                    output_file.fp.write(json.dumps(x, default=bson.json_util.default, ensure_ascii=False) + '\n')
                logger.info(progress)
                logger.info(f"Saved {len(writer)} items to [{writer.opt}]")

        return cls.app


app.add_typer(ExtractApp.typer(), name="extract")
app.add_typer(ExportApp.typer(), name="export")

if __name__ == "__main__":
    app()
