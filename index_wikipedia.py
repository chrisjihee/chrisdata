import json
import logging
import math
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Iterable
from elastic_transport import ObjectApiResponse
from elasticsearch.helpers import streaming_bulk

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from more_itertools import ichunked
from pymongo.collection import Collection

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments, TableOption, IndexOption, MongoDBTable, ElasticSearchClient
from chrisbase.io import LoggingFormat, iter_compressed
from chrisbase.util import to_dataframe, mute_tqdm_cls
from crawl_wikipedia import ProcessResult as WikipediaProcessResult

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class DataOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    total: int = field(default=9740173)  # wc -l Wikipedia-20230920-parse-kowiki.jsonl
    start: int = field(default=0)
    limit: int = field(default=-1)
    batch: int = field(default=1)
    from_table: bool = field(default=False)
    prog_interval: int = field(default=10000)

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)


@dataclass
class ProgramArguments(CommonArguments):
    data: DataOption = field()
    table: TableOption = field()
    index: IndexOption = field()
    other: str | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
            to_dataframe(columns=columns, raw=self.index, data_prefix="index"),
            to_dataframe(columns=columns, raw={"other": self.other}),
        ]).reset_index(drop=True)


@dataclass
class PassageUnit(DataClassJsonMixin):
    _id: str
    title: str
    subtitle1: str
    subtitle2: str
    body_text: str


def process_one(x: str, processed: set[str]) -> Iterable[PassageUnit]:
    doc: WikipediaProcessResult = WikipediaProcessResult.from_json(x)
    if not doc.title or not doc.page_id or not doc.section_list:
        return None
    doc.title = doc.title.strip()
    if doc.title in processed:
        return None
    sect_ids: tuple[int, int] = (1, 1)
    sect_heads: tuple[str, str] = ("", "")
    sect_texts_prev: list[str] = []
    for (_, h1, h2, sect_body) in doc.section_list:
        h1, h2 = h1.strip(), h2.strip()
        sect_texts = [x for x in [x.strip() for x in sect_body.strip().splitlines()] if len(x) > 0]
        if sect_heads[0] != h1.strip():
            sect_ids = (sect_ids[0] + 1, 1)
            sect_heads = (h1, h2)
            sect_texts_prev = sect_texts
        elif sect_heads[1] != h2.strip():
            sect_ids = (sect_ids[0], sect_ids[1] + 1)
            sect_heads = (h1, h2)
            sect_texts_prev = sect_texts
        elif sect_texts_prev != sect_texts:
            sect_ids = (sect_ids[0], sect_ids[1] + 1)
            sect_heads = (h1, h2)
            sect_texts_prev = sect_texts
        else:
            continue
        for text_id, text in enumerate(sect_texts, start=1):
            path_ids = sect_ids + (text_id,)
            _id = f"{doc.title}-{'-'.join([f'{i:03d}' for i in path_ids])}"
            yield PassageUnit(_id=_id, title=doc.title, subtitle1=h1, subtitle2=h2, body_text=text)
    processed.add(doc.title)


def process_many(batch: Iterable[str], table: Collection, processed: set[str]):
    batch_units = [x for x in [process_one(x, processed) for x in batch] if x]
    all_units = [unit for batch in batch_units for unit in batch]
    rows = [row.to_dict() for row in all_units if row]
    if len(rows) > 0:
        table.insert_many(rows)


@app.command()
def index(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="index_wikipedia"),
        output_home: str = typer.Option(default="output-index_wikipedia"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # data
        input_home: str = typer.Option(default="input/Wikidata-parse"),
        input_name: str = typer.Option(default="Wikipedia-20230920-parse-kowiki.jsonl.bz2"),
        input_total: int = typer.Option(default=9740173),
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=50),
        input_batch: int = typer.Option(default=3),
        from_table: bool = typer.Option(default=False),
        prog_interval: int = typer.Option(default=10),
        # table
        db_host: str = typer.Option(default="localhost:6382"),
        tab_name: str = typer.Option(default="parse_wikipedia"),
        # index
        index_host: str = typer.Option(default="localhost:9200"),
        index_user: str = typer.Option(default="elastic"),
        index_pswd: str = typer.Option(default="cfg/elastic-pw.txt"),
        index_cert: str = typer.Option(default="elasticsearch/config/certs/http_ca.crt"),
        index_name: str = typer.Option(default="wikipedia-20230920-index-kowiki"),
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
    args = ProgramArguments(
        env=env,
        data=DataOption(
            home=input_home,
            name=input_name,
            total=input_total,
            start=input_start,
            limit=input_limit,
            batch=input_batch,
            from_table=from_table,
            prog_interval=prog_interval,
        ),
        table=TableOption(
            db_host=db_host,
            db_name=env.project,
            tab_name=tab_name,
        ),
        index=IndexOption(
            host=index_host,
            user=index_user,
            pswd=index_pswd,
            cert=index_cert,
            name=index_name,
        ),
    )
    tqdm = mute_tqdm_cls()
    # output_name = args.data.name.stem.replace("-parse-", "-index-").replace(".jsonl", "")
    # output_file = (args.env.output_home / f"{output_name}-{args.env.time_stamp}.jsonl")

    with (JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='=')):
        with MongoDBTable(args.table) as inp_table, ElasticSearchClient(args.index) as out_client:
            if args.data.from_table:
                logger.info(f"Use database table: {args.table}")
                find_opt = {}
                num_input, inputs = inp_table.count_documents(find_opt), inp_table.find(find_opt).sort("_id")
            else:
                num_input, inputs = args.data.total, iter_compressed(args.data.home / args.data.name)
                inputs = map(PassageUnit.to_dict, map(PassageUnit.from_json, inputs))
            if args.data.start > 0:
                num_input, inputs = max(0, min(num_input, num_input - args.data.start)), islice(inputs, args.data.start, num_input)
            if args.data.limit > 0:
                num_input, inputs = min(num_input, args.data.limit), islice(inputs, args.data.limit)
            num_batch, batches = math.ceil(num_input / args.data.batch), ichunked(inputs, args.data.batch)
            logger.info(f"Index {num_input} inputs with {num_batch} batches to {args.index}")
            progress, interval = (tqdm(batches, total=num_batch, unit="batch", pre="*", desc="indexing"),
                                  math.ceil(args.data.prog_interval / args.data.batch))

            if out_client.indices.exists(index=args.index.name):
                logger.info(f"Delete existing index: {args.index}")
                out_client.indices.delete(index=args.index.name)
            logger.info(f"Creating an index: {args.index}")
            out_client.indices.create(
                index=args.index.name,
                settings={
                    "analysis": {
                        "analyzer": {"my_analyzer": {"tokenizer": "my_tokenizer"}},
                        "tokenizer": {"my_tokenizer": {"type": "ngram", "min_gram": "2", "max_gram": "2", }},
                    },
                    "index": {
                        "number_of_shards": 1,
                        "blocks": {"read_only_allow_delete": "false"},
                    }
                },
                mappings={
                    "properties": {
                        "index": {"type": "integer"},
                        "id": {"type": "keyword"},
                        "lang": {"type": "keyword"},
                        "date": {"type": "keyword"},
                        "title": {"type": "keyword"},
                        "body": {"type": "text", "analyzer": "my_analyzer"},
                        "both": {"type": "text", "analyzer": "my_analyzer"},
                    },
                },
            )
            logger.info(f"Created an index: {args.index}")

            def _document_generator(documents_path):
                with open(documents_path) as fp:
                    documents = list(map(lambda x: json.loads(x), fp.readlines()))
                    for i, document in enumerate(documents):
                        yield dict(
                            # index=i,
                            id=document["id"],
                            lang=document["lang"],
                            date=document["date"],
                            title=document["title"],
                            body=document['body'],
                            both=f"{document['title']} {document['body']}"
                        )

            def _input_generator(batch):
                for xxx in batch:
                    logger.info(f"Index a document: {xxx}")
                    yield xxx

            for i, x in enumerate(progress):
                if i > 0 and i % interval == 0:
                    logger.info(progress)
                # process_many(batch=x, table=out_table, processed=processed)
                for ok, action in streaming_bulk(client=out_client,
                                                 # actions=x,
                                                 actions=_input_generator(batch=x),
                                                 # actions=_document_generator(documents_path="input/CHOSUN_2000.small.jsonl"),
                                                 index=args.index.name,
                                                 chunk_size=256, ):
                    logger.info(f"ok={ok}, action={action}")
            logger.info(progress)
            out_client.indices.refresh(index=args.index.name)
            for line in str(out_client.cat.indices(index=args.index.name, v=True).body.strip()).splitlines():
                logger.info(line)

            query, nbest = '카터 * 대한민국', 10
            response: ObjectApiResponse = out_client.search(
                index=args.index.name,
                query={
                    "match": {
                        "body_text": {"query": query}
                    },
                },
                _source=("_id", "title", "subtitle1", "subtitle2", "body_text"),
                size=nbest,
            )
            response = dict(response)
            logger.info(response["hits"])
            logger.info(response["hits"]["total"])
            logger.info(response["hits"]["max_score"])
            for hit in response["hits"]["hits"]:
                logger.info(f"Retrieve a document: {hit}")
            logger.info(response["hits"].keys())
            # logger.info(json.dumps(response, ensure_ascii=False))

        #     find_opt = {}
        #     num_row, rows = out_table.count_documents(find_opt), out_table.find(find_opt).sort("_id")
        #     progress, interval = (tqdm(rows, total=num_row, unit="row", pre="*", desc="exporting"),
        #                           args.data.prog_interval * 100)
        #     for i, x in enumerate(progress):
        #         if i > 0 and i % interval == 0:
        #             logger.info(progress)
        #         out_file.write(json.dumps(x, ensure_ascii=False) + '\n')
        #     logger.info(progress)
        # logger.info(f"Export {num_row} rows to {output_file}")


if __name__ == "__main__":
    app()
