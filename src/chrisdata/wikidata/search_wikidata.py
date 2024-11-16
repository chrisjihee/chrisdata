import json
from pathlib import Path
from typing import Iterable

import bson.json_util
import pandas as pd
import typer
from elasticsearch.helpers import streaming_bulk

from chrisbase.data import InputOption, OutputOption, FileOption, TableOption, IndexOption
from chrisbase.data import JobTimer, ProjectEnv, OptionData
from chrisbase.data import Streamer, FileStreamer, ElasticStreamer
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls
from . import *

logger = logging.getLogger(__name__)
app = AppTyper()


class SearchApp:
    app = AppTyper()

    @dataclass
    class FilterOption(OptionData):
        min_char: int = field()
        max_char: int = field()
        max_word: int = field()
        min_hits: int = field()
        max_hits: int = field()
        min_cooccur: int = field()
        black_prop: str | Path = field()
        num_black_prop: int = 0
        set_black_prop = set()
        parenth_ending = re.compile(r" \(.+?\)$")
        korean_starting = re.compile(r"^[가-힣].+")

        def __post_init__(self):
            self.black_prop = Path(self.black_prop)
            if self.black_prop.exists() and self.black_prop.is_file():
                lines = (x.strip() for x in self.black_prop.read_text().splitlines())
                self.set_black_prop = {x for x in lines if x}
                self.num_black_prop = len(self.set_black_prop)

        def invalid_prop(self, x: str) -> bool:
            return x in self.set_black_prop

        def normalize_title(self, x: str) -> str:
            return self.parenth_ending.sub("", x).strip()

        def invalid_title(self, full: str, norm: str) -> bool:
            return (
                    len(norm) < self.min_char or
                    len(norm) > self.max_char or
                    len(norm.split()) > self.max_word or
                    not self.korean_starting.match(norm) or
                    norm.startswith("위키백과:") or norm.startswith("분류:") or norm.startswith("포털:") or
                    norm.startswith("틀:") or norm.startswith("모듈:") or norm.startswith("위키프로젝트:") or
                    full.endswith("(동음이의)")
            )

        def invalid_entity(self, e: EntityInWiki):
            return e.hits < self.min_hits or self.max_hits < e.hits

        def invalid_triple(self, p: TripleInWiki):
            return p.hits < self.min_cooccur

    @classmethod
    def typer(cls) -> typer.Typer:

        @dataclass
        class SearchArguments(IOArguments):
            filter: cls.FilterOption = field()

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
                    to_dataframe(columns=columns, raw=self.output, data_prefix="input", data_exclude=["file", "table", "index"]),
                    to_dataframe(columns=columns, raw=self.output.file, data_prefix="output.file") if self.output.file else None,
                    to_dataframe(columns=columns, raw=self.output.table, data_prefix="output.table") if self.output.table else None,
                    to_dataframe(columns=columns, raw=self.output.index, data_prefix="output.index") if self.output.index else None,
                    to_dataframe(columns=columns, raw=self.filter, data_prefix="filter"),
                ]).reset_index(drop=True)

        def search_query(query: str, input_index: ElasticStreamer):
            try:
                response = input_index.cli.search(
                    index=input_index.opt.name,
                    query={
                        "query_string": {
                            "default_field": "body_text",
                            "query": query
                        },
                    },
                    _source=("_id", "title", "subtitle1", "subtitle2", "body_text"),
                    # size=10000,
                )
                if response.meta.status == 200:
                    body = response.body
                    max_score = body['hits']['max_score']
                    num_hits = body['hits']['total']['value']
                    return num_hits, max_score
                return 0, 0.0
            except Exception as e:
                logger.error(f"Failed to search [{query}] from [{input_index.opt}]")
                logger.error(e)
                exit(1)

        def escape_query(q: str) -> str:
            special_characters = [
                '\\', '+', '-', '=', '&&', '||', '>', '<', '!', '(', ')',
                '{', '}', '[', ']', '^', '"', '~', '*', '?', ':'
            ]
            for char in special_characters:
                q = q.replace(char, f"\\{char}")
            return q

        def search_each(query: str, input_index: ElasticStreamer):
            query = escape_query(query)
            return search_query(f'"{query}"', input_index)

        def search_both(query1: str, query2: str, input_index: ElasticStreamer):
            query1 = escape_query(query1)
            query2 = escape_query(query2)
            return search_query(f'"{query1}" AND "{query2}"', input_index)

        def search_one(x: dict, input_table: MongoStreamer, input_index: ElasticStreamer,
                       opt: cls.FilterOption, invalid_queries: set[str], relation_cache: dict[str, Relation], debug: bool = False):
            wikidata_item = WikidataUnit.from_dict(x)
            if wikidata_item.type == "item":
                subject_full = wikidata_item.title1
                subject_norm = opt.normalize_title(subject_full)
                object_norms = set()
                if not opt.invalid_title(full=subject_full, norm=subject_norm) and subject_norm not in invalid_queries:
                    subject_ex = EntityInWiki(subject_norm, *search_each(subject_norm, input_index))
                    if opt.invalid_entity(subject_ex):
                        invalid_queries.add(subject_norm)
                    else:
                        for claim in wikidata_item.claims:
                            relation_id = claim['property']
                            if relation_id not in relation_cache:
                                r = input_table.table.find_one({'_id': relation_id})
                                relation = Relation(relation_id, r['label1'], r['label2'])
                                relation_cache[relation_id] = relation
                            if not opt.invalid_prop(relation_id):
                                value = claim['datavalue']
                                value_type = value['type']
                                if value_type == "wikibase-entityid":
                                    entity_type = value['value']['entity-type']
                                    if entity_type == "item":
                                        item_id = value['value']['id']
                                        item_row = input_table.table.find_one({'_id': item_id})
                                        if item_row:
                                            object_full = item_row['title1']
                                            object_norm = opt.normalize_title(object_full)
                                            if subject_norm != object_norm and object_norm not in subject_norm and subject_norm not in object_norm:
                                                if not opt.invalid_title(full=object_full, norm=object_norm) and object_norm not in invalid_queries:
                                                    object_norms.add(object_norm)
                                                    object_ex = EntityInWiki(object_norm, *search_each(object_norm, input_index))
                                                    if opt.invalid_entity(object_ex):
                                                        invalid_queries.add(object_norm)
                                                    else:
                                                        triple_ex = TripleInWiki(subject_ex, object_ex, relation_cache[relation_id],
                                                                                 *search_both(subject_ex.entity, object_ex.entity, input_index))
                                                        if debug:
                                                            logger.info(f"  - {subject_ex} -- {relation_cache[relation_id]} -- {object_ex}")
                                                            logger.info(f"    => {triple_ex}")
                                                        if not opt.invalid_triple(triple_ex):
                                                            yield triple_ex

        def search_many(batch: Iterable[dict], writer: Streamer,
                        input_table: MongoStreamer, input_index: ElasticStreamer,
                        filter_opt: cls.FilterOption, invalid_queries: set[str], relation_cache: dict[str, Relation], debug: bool = False):
            batch_units = [
                list(x) for x in [
                    search_one(x, input_table, input_index, opt=filter_opt,
                               invalid_queries=invalid_queries, relation_cache=relation_cache)
                    for x in batch] if x
            ]
            all_units = [unit for batch in batch_units for unit in batch]
            if debug:
                logger.info(f"* All Units({len(all_units)}):")
                for a in all_units:
                    logger.info(f"  - {a}")
            rows = [row.to_dict() for row in all_units if row]
            if len(rows) > 0:
                if isinstance(writer, MongoStreamer):
                    writer.table.insert_many(rows)
                elif isinstance(writer, ElasticStreamer):
                    for ok, action in streaming_bulk(writer.cli, actions=rows, chunk_size=len(rows), index=writer.opt.name, yield_ok=False):
                        logger.warning(f"ok={ok}, action={action}")
                else:
                    raise ValueError(f"Unsupported writer: {type(writer)}")

        @cls.app.command()
        def run(
                # env
                project: str = typer.Option(default="chrisdata"),
                job_name: str = typer.Option(default="search_wikidata"),
                output_home: str = typer.Option(default="output-search_wikidata"),
                logging_file: str = typer.Option(default="search.out"),
                debugging: bool = typer.Option(default=False),
                # input
                input_start: int = typer.Option(default=0),
                input_limit: int = typer.Option(default=-1),
                input_batch: int = typer.Option(default=1000),
                input_inter: int = typer.Option(default=5000),
                input_index_home: str = typer.Option(default="localhost:9810"),
                input_index_name: str = typer.Option(default="wikipedia-20230920-index-kowiki"),
                input_index_user: str = typer.Option(default="elastic"),
                input_index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
                input_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
                input_table_name: str = typer.Option(default="wikidata-20230920-parse-kowiki"),
                # output
                output_index_home: str = typer.Option(default="localhost:9810"),
                output_index_name: str = typer.Option(default="wikidata-20230920-search-kowiki"),
                output_index_user: str = typer.Option(default="elastic"),
                output_index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
                output_index_reset: bool = typer.Option(default=False),
                output_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
                output_table_name: str = typer.Option(default="wikidata-20230920-search-kowiki"),
                output_table_reset: bool = typer.Option(default=False),
                # filter
                filter_min_char: int = typer.Option(default=2),
                filter_max_char: int = typer.Option(default=20),
                filter_max_word: int = typer.Option(default=5),
                filter_min_hits: int = typer.Option(default=5),
                filter_max_hits: int = typer.Option(default=1000),
                filter_min_cooccur: int = typer.Option(default=3),  # TODO: filter_min_cooccur를 올려보기: 2, 3, ...?
                filter_black_prop: str = typer.Option(default="input/wikimedia/wikidata-black_prop-x.txt"),
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
                ),
                table=TableOption(
                    home=input_table_home,
                    name=input_table_name,
                ),
            )
            output_opt = OutputOption(
                index=IndexOption(
                    home=output_index_home,
                    user=output_index_user,
                    pswd=output_index_pswd,
                    name=output_index_name,
                    reset=output_index_reset,
                ),
                table=TableOption(
                    home=output_table_home,
                    name=output_table_name,
                    reset=output_table_reset,
                ),
            )
            filter_opt = cls.FilterOption(
                min_char=filter_min_char,
                max_char=filter_max_char,
                max_word=filter_max_word,
                min_hits=filter_min_hits,
                max_hits=filter_max_hits,
                min_cooccur=filter_min_cooccur,
                black_prop=filter_black_prop,
            )
            args = SearchArguments(
                env=env,
                input=input_opt,
                output=output_opt,
                filter=filter_opt,
            )
            tqdm = mute_tqdm_cls()
            assert args.input.index, "input.index is required"
            assert args.input.table, "input.table is required"
            assert args.output.index or args.output.table, "output.index or output.table is required"
            logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)

            with (
                JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
                ElasticStreamer(args.output.index) as output_index, MongoStreamer(args.output.table) as output_table,
                ElasticStreamer(args.input.index) as input_index, MongoStreamer(args.input.table) as input_table,
            ):
                # search parsed data
                writer = Streamer.first_usable(output_index, output_table)
                reader = Streamer.first_usable(input_table)
                input_items: InputOption.InputItems = args.input.ready_inputs(reader, len(reader))
                logger.info(f"Run SearchApp")
                logger.info(f"- from: [{type(reader).__name__}] [{reader.opt}]({len(reader)})")
                logger.info(f"  => amount: {input_items.num_item}{'' if input_items.has_single_items() else f' * {args.input.batch}'} ({type(input_items).__name__})")
                logger.info(f"- with: [{type(input_index).__name__}] [{input_index.opt}]({len(input_index)})")
                logger.info(f"  => filter: set_black_prop={args.filter.set_black_prop}, ...")  # TODO: Bridge Entity가 없으면 black_prop를 줄여보자!
                logger.info(f"- into: [{type(writer).__name__}] [{writer.opt}]({len(writer)})")
                progress, interval = (
                    tqdm(input_items.items, total=input_items.num_item, unit="batch", pre="*", desc="searching"),
                    math.ceil(args.input.inter / args.input.batch)
                )
                relation_cache = dict()
                invalid_queries = set()
                for i, batch in enumerate(progress):
                    if i > 0 and i % interval == 0:
                        logger.info(progress)
                    search_many(batch=batch, writer=writer, input_table=input_table, input_index=input_index,
                                filter_opt=args.filter, invalid_queries=invalid_queries, relation_cache=relation_cache)
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
                job_name: str = typer.Option(default="search_wikidata"),
                output_home: str = typer.Option(default="output-search_wikidata"),
                logging_file: str = typer.Option(default="export.out"),
                debugging: bool = typer.Option(default=False),
                # input
                input_batch: int = typer.Option(default=1),
                input_inter: int = typer.Option(default=5000),
                input_index_home: str = typer.Option(default="localhost:9810"),
                input_index_name: str = typer.Option(default="wikidata-20230920-search-kowiki"),
                input_index_user: str = typer.Option(default="elastic"),
                input_index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
                input_index_sort: str = typer.Option(default="hits:desc"),
                input_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
                input_table_name: str = typer.Option(default="wikidata-20230920-search-kowiki"),
                # output
                output_file_home: str = typer.Option(default="output-search_wikidata"),
                output_file_name: str = typer.Option(default="wikidata-20230920-search-kowiki-new.jsonl"),
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
                    sort=input_index_sort,
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
            logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
            assert args.input.index or args.input.table, "input.index or input.table is required"
            assert args.output.file, "output.file is required"

            with (
                JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
                ElasticStreamer(args.input.index) as input_index, MongoStreamer(args.input.table) as input_table,
                FileStreamer(args.output.file) as output_file,
            ):
                # export search results
                writer = Streamer.first_usable(output_file)
                reader = Streamer.first_usable(input_index, input_table)
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


app.add_typer(SearchApp.typer(), name="search")
app.add_typer(ExportApp.typer(), name="export")

if __name__ == "__main__":
    app()
