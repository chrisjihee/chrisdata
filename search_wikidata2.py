import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, CommonArguments, OptionData, TypedData
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption, IndexOption
from chrisbase.data import LineFileWrapper, MongoDBWrapper, ElasticSearchWrapper
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls
from parse_wikidata import WikidataUnit

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class EntityInWiki(TypedData):
    entity: str
    hits: int
    score: float


@dataclass
class EntityPairInWiki(TypedData):
    entity1: EntityInWiki
    entity2: EntityInWiki
    hits: int
    score: float = field(default=0.0)
    pmi: float = field(default=0.0)

    @staticmethod
    def calc_pmi(h_xy: float, h_x: float, h_y: float, n: int = 10000, e: float = 0.0000001) -> float:
        p_xy = h_xy / n
        p_x = h_x / n
        p_y = h_y / n
        return math.log2((p_xy + e) / ((p_x * p_y) + e))

    def __post_init__(self):
        if self.score is None:
            self.score = 0.0
        self.pmi = self.calc_pmi(self.hits, self.entity1.hits, self.entity2.hits)


@dataclass
class FilterOption(OptionData):
    min_char: int = field()
    max_char: int = field()
    max_word: int = field()
    min_hits: int = field()
    max_hits: int = field()
    min_score: float = field()
    max_score: float = field()
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
        return (
                e.hits < self.min_hits or
                e.hits > self.max_hits or
                e.score < self.min_score or
                e.score > self.max_score
        )

    def invalid_cooccur(self, p: EntityPairInWiki):
        return (
                p.hits < self.min_cooccur
        )


@dataclass
class SearchArguments(CommonArguments):
    input: InputOption = field()
    output: OutputOption = field()
    filter: FilterOption = field()

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
            to_dataframe(columns=columns, raw=self.filter, data_prefix="filter"),
        ]).reset_index(drop=True)


def search_query(query: str, input_index: ElasticSearchWrapper):
    query = query.replace('""', '"')
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
        # for hit in body["hits"]["hits"]:
        #     logger.info(f"  - {hit}")
        # return body['hits']['total']['value']
        return num_hits, max_score
    return 0, 0.0


def search_each(query: str, input_index: ElasticSearchWrapper):
    return search_query(f'"{query}"', input_index)


def search_both(query1: str, query2: str, input_index: ElasticSearchWrapper):
    return search_query(f'"{query1}" AND "{query2}"', input_index)


def search_one(x: dict, input_table: MongoDBWrapper, input_index: ElasticSearchWrapper, opt: FilterOption, invalid_queries: set[str]):
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
                    prop_id = claim['property']
                    if not opt.invalid_prop(prop_id):
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
                                    if subject_norm != object_norm and not opt.invalid_title(full=object_full, norm=object_norm) and object_norm not in invalid_queries:
                                        # print(f"- {prop_id:100s} ====> \t\t\t{entity_id:10s} -> {object_norm}")
                                        object_norms.add(object_norm)
                valid_objects = list()
                for object_norm in object_norms:
                    object_ex = EntityInWiki(object_norm, *search_each(object_norm, input_index))
                    if opt.invalid_entity(object_ex):
                        invalid_queries.add(object_norm)
                    else:
                        valid_objects.append(object_ex)
                if len(valid_objects) > 0:
                    valid_pairs = list()
                    # sorted_objects = sorted(valid_objects, key=operator.attrgetter('score'), reverse=True)
                    # logger.info(f"subject_ex={subject_ex}")
                    # logger.info(f"sorted_objects={sorted_objects}")
                    for object_ex in valid_objects:
                        pair_ex = EntityPairInWiki(subject_ex, object_ex,
                                                   *search_both(subject_ex.entity, object_ex.entity, input_index))
                        if not opt.invalid_cooccur(pair_ex):
                            valid_pairs.append(pair_ex)
                    # sorted_pairs = sorted(valid_pairs, key=operator.attrgetter('score'), reverse=True)
                    # logger.info(f"sorted_pairs={sorted_pairs}")
                    # logger.info(f"")
                    for pair in valid_pairs:
                        yield pair


def search_many(batch: Iterable[dict], output_table: MongoDBWrapper, input_table: MongoDBWrapper, input_index: ElasticSearchWrapper, filter_opt: FilterOption, invalid_queries: set[str]):
    batch_units = [x for x in [search_one(x, input_table, input_index, opt=filter_opt, invalid_queries=invalid_queries) for x in batch] if x]
    all_units = [unit for batch in batch_units for unit in batch]
    # sorted_units = sorted(all_units, key=operator.attrgetter('pmi'), reverse=True)
    rows = [row.to_dict() for row in all_units if row]
    if len(rows) > 0:
        output_table.table.insert_many(rows)


@app.command()
def search(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="search_wikidata"),
        output_home: str = typer.Option(default="output-search_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),
        # input_limit: int = typer.Option(default=20000),
        input_batch: int = typer.Option(default=1000),
        input_inter: int = typer.Option(default=5000),
        input_total: int = typer.Option(default=1018174),
        input_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        input_table_name: str = typer.Option(default="wikidata-20230920-parse-kowiki"),
        input_index_home: str = typer.Option(default="localhost:9810"),
        input_index_name: str = typer.Option(default="wikipedia-20230920-index-kowiki"),
        input_index_user: str = typer.Option(default="elastic"),
        input_index_pswd: str = typer.Option(default="cIrEP5OCwTLn0QIQwnsA"),
        # output
        output_file_home: str = typer.Option(default="input/wikimedia"),
        output_file_name: str = typer.Option(default="wikidata-20230920-search-kowiki-new.jsonl"),
        output_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        output_table_name: str = typer.Option(default="wikidata-20230920-search-kowiki"),
        output_table_reset: bool = typer.Option(default=True),
        # filter
        filter_min_char: int = typer.Option(default=2),
        filter_max_char: int = typer.Option(default=20),
        filter_max_word: int = typer.Option(default=5),
        filter_min_hits: int = typer.Option(default=10),
        filter_max_hits: int = typer.Option(default=1000),
        filter_min_score: float = typer.Option(default=0.5),
        filter_max_score: float = typer.Option(default=100.0),
        filter_min_cooccur: int = typer.Option(default=1),
        filter_black_prop: str = typer.Option(default="input/wikimedia/wikidata-black_prop.txt"),
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
        table=TableOption(
            home=input_table_home,
            name=input_table_name,
            strict=True,
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
            mode="w",
        ) if output_file_home and output_file_name else None,
        table=TableOption(
            home=output_table_home,
            name=output_table_name,
            reset=output_table_reset,
        ) if output_table_home and output_table_name else None,
    )
    filter_opt = FilterOption(
        min_char=filter_min_char,
        max_char=filter_max_char,
        max_word=filter_max_word,
        min_hits=filter_min_hits,
        max_hits=filter_max_hits,
        min_score=filter_min_score,
        max_score=filter_max_score,
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
    logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
    assert args.input.index, "input.index is required"
    assert args.input.file or args.input.table, "input.file or input.table is required"
    assert args.output.file, "output.file is required"
    assert args.output.table, "output.table is required"

    with (
        JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoDBWrapper(args.output.table) as output_table, LineFileWrapper(args.output.file) as output_file,
        MongoDBWrapper(args.input.table) as input_table,
        ElasticSearchWrapper(args.input.index) as input_index,
    ):
        # search parsed data
        inputs = args.input.select_inputs(input_table)
        outputs = args.output.select_outputs(output_table, output_file)
        logger.info(f"Search from [{inputs.wrapper.opt}] with [{args.input.index}] to [{outputs.wrapper.opt}]")
        logger.info(f"- amount: inputs={inputs.num_input}, batches={inputs.num_batch}")
        logger.info(f"- filter: num_black_sect={args.filter.num_black_prop}, min_char={args.filter.min_char}, max_char={args.filter.max_char}, max_word={args.filter.max_word}")
        progress, interval = (
            tqdm(inputs.batches, total=inputs.num_batch, unit="batch", pre="*", desc="searching"),
            math.ceil(args.input.inter / args.input.batch)
        )
        invalid_queries = set()
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            search_many(batch=x, output_table=output_table, input_table=input_table, input_index=input_index, filter_opt=args.filter, invalid_queries=invalid_queries)
        logger.info(progress)
        # logger.info(f"Indexed {len(data_index)} documents to [{args.data.index}]")


if __name__ == "__main__":
    app()
