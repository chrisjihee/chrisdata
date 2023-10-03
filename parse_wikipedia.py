import json
import logging
import math
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from more_itertools import ichunked
from pymongo.collection import Collection

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments, TableOption, MongoDBTable
from chrisbase.io import LoggingFormat, iter_compressed
from chrisbase.util import to_dataframe, mute_tqdm_cls
from crawl_wikipedia import ProcessResult as WikipediaProcessResult

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class DataOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    total: int = field(default=1410203)  # wc -l Wikipedia-20230920-crawl-kowiki.jsonl
    start: int = field(default=0)
    limit: int = field(default=-1)
    batch: int = field(default=1)
    from_scratch: bool = field(default=False)
    prog_interval: int = field(default=10000)

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)


@dataclass
class ProgramArguments(CommonArguments):
    data: DataOption = field()
    table: TableOption = field()
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
            to_dataframe(columns=columns, raw={"other": self.other}),
        ]).reset_index(drop=True)


@dataclass
class PassageUnit(DataClassJsonMixin):
    _id: str
    title: str
    subtitle1: str
    subtitle2: str
    body_text: str


def process_one(x: str, processed: set[int]) -> Iterable[PassageUnit]:
    doc: WikipediaProcessResult = WikipediaProcessResult.from_json(x)
    doc.title = doc.title.strip() if doc.title else ""
    if not doc.page_id or doc.page_id in processed or not doc.title or not doc.section_list:
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
            _id = f"{doc.page_id:07d}-{'-'.join([f'{i:03d}' for i in path_ids])}"
            yield PassageUnit(_id=_id, title=doc.title, subtitle1=h1, subtitle2=h2, body_text=text)
    processed.add(doc.page_id)


def process_many(batch: Iterable[str], table: Collection, processed: set[int]):
    batch_units = [x for x in [process_one(x, processed) for x in batch] if x]
    all_units = [unit for batch in batch_units for unit in batch]
    rows = [row.to_dict() for row in all_units if row]
    if len(rows) > 0:
        table.insert_many(rows)


@app.command()
def parse(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="parse_wikipedia"),
        output_home: str = typer.Option(default="output-parse_wikipedia"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # data
        input_home: str = typer.Option(default="input/Wikidata-parse"),
        input_name: str = typer.Option(default="Wikipedia-20230920-crawl-kowiki.jsonl.bz2"),
        input_total: int = typer.Option(default=1410203),
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=1000),
        prog_interval: int = typer.Option(default=10000),
        from_scratch: bool = typer.Option(default=True),
        # table
        db_host: str = typer.Option(default="localhost:6382"),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        output_home=output_home,
        logging_file=logging_file,
        msg_level=logging.DEBUG if debugging else logging.INFO,
        msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
    )
    data_opt = DataOption(
        home=input_home,
        name=input_name,
        total=input_total,
        start=input_start,
        limit=input_limit,
        batch=input_batch,
        from_scratch=from_scratch,
        prog_interval=prog_interval,
    )
    table_name = data_opt.name.stem.replace("-crawl-", "-parse-").replace(".jsonl", "")
    table_opt = TableOption(
        db_host=db_host,
        db_name=env.project,
        tab_name=table_name,
    )
    args = ProgramArguments(
        env=env,
        data=data_opt,
        table=table_opt,
    )
    tqdm = mute_tqdm_cls()
    output_file = (env.output_home / f"{table_name}-{env.time_stamp}.jsonl")

    with (JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='=')):
        with MongoDBTable(args.table) as out_table, output_file.open("w") as out_file:
            if args.data.from_scratch:
                logger.info(f"Clear database table: {args.table}")
                out_table.drop()
            num_input, inputs = args.data.total, iter_compressed(args.data.home / args.data.name)
            if args.data.start > 0:
                num_input, inputs = max(0, min(num_input, num_input - args.data.start)), islice(inputs, args.data.start, num_input)
            if args.data.limit > 0:
                num_input, inputs = min(num_input, args.data.limit), islice(inputs, args.data.limit)
            num_batch, batches = math.ceil(num_input / args.data.batch), ichunked(inputs, args.data.batch)
            logger.info(f"Parse {num_input} inputs with {num_batch} batches")
            progress, interval = (tqdm(batches, total=num_batch, unit="batch", pre="*", desc="importing"),
                                  math.ceil(args.data.prog_interval / args.data.batch))

            processed = set()
            for i, x in enumerate(progress):
                if i > 0 and i % interval == 0:
                    logger.info(progress)
                process_many(batch=x, table=out_table, processed=processed)
            logger.info(progress)
            find_opt = {}
            num_row, rows = out_table.count_documents(find_opt), out_table.find(find_opt).sort("_id")
            progress, interval = (tqdm(rows, total=num_row, unit="row", pre="*", desc="exporting"),
                                  args.data.prog_interval * 100)
            for i, x in enumerate(progress):
                if i > 0 and i % interval == 0:
                    logger.info(progress)
                out_file.write(json.dumps(x, ensure_ascii=False) + '\n')
            logger.info(progress)
        logger.info(f"Export {num_row} rows to {output_file}")


if __name__ == "__main__":
    app()
