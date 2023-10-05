import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, CommonArguments, OptionData
from chrisbase.data import InputOption, FileOption, TableOption
from chrisbase.data import LineFileWrapper, MongoDBWrapper
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls
from crawl_wikipedia import ProcessResult as WikipediaProcessResult

logger = logging.getLogger(__name__)
app = AppTyper()


@dataclass
class FilterOption(OptionData):
    min_char: int = field(default=40)
    min_word: int = field(default=5)
    black_sect: str | Path = field(default="black_sect.txt")
    num_black_sect: int = 0
    set_black_sect = set()

    def __post_init__(self):
        self.black_sect = Path(self.black_sect)
        if self.black_sect.exists() and self.black_sect.is_file():
            lines = (x.strip() for x in self.black_sect.read_text().splitlines())
            self.set_black_sect = {x for x in lines if x}
            self.num_black_sect = len(self.set_black_sect)

    def invalid_sect(self, x: str) -> bool:
        return x in self.set_black_sect

    def valid_text(self, x: str) -> bool:
        return all((
            '다.' in x,
            len(x) > self.min_char,
            len(str(x).split()) >= self.min_word,
        ))


@dataclass
class ParseArguments(CommonArguments):
    data: InputOption = field()
    filter: FilterOption = field(default=FilterOption())

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
            to_dataframe(columns=columns, raw=self.filter, data_prefix="filter"),
        ]).reset_index(drop=True)


@dataclass
class PassageUnit(DataClassJsonMixin):
    _id: str
    title: str
    subtitle1: str
    subtitle2: str
    body_text: str


def process_one(x: str, parsed_ids: set[int], opt: FilterOption = FilterOption()) -> Iterable[PassageUnit]:
    doc: WikipediaProcessResult = WikipediaProcessResult.from_json(x)
    doc.title = doc.title.strip() if doc.title else ""
    if not doc.page_id or doc.page_id in parsed_ids or not doc.title or not doc.section_list:
        return None
    sect_ids: tuple[int, int] = (1, 1)
    sect_heads: tuple[str, str] = ("", "")
    sect_texts_prev: list[str] = []
    for (_, h1, h2, sect_body) in doc.section_list:
        h1, h2 = h1.strip(), h2.strip()
        if opt.invalid_sect(h1):
            continue
        sect_lines = [x.strip() for x in sect_body.strip().splitlines()]
        sect_texts = [x for x in sect_lines if opt.valid_text(x)]
        if not sect_texts:
            continue
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
    parsed_ids.add(doc.page_id)


def parse_many(batch: Iterable[str], wrapper: MongoDBWrapper, parsed_ids: set[int]):
    batch_units = [x for x in [process_one(x, parsed_ids) for x in batch] if x]
    all_units = [unit for batch in batch_units for unit in batch]
    rows = [row.to_dict() for row in all_units if row]
    if len(rows) > 0:
        wrapper.table.insert_many(rows)


@app.command()
def parse(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="parse_wikipedia"),
        output_home: str = typer.Option(default="output-parse_wikipedia"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # data
        data_start: int = typer.Option(default=0),
        data_limit: int = typer.Option(default=-1),
        data_batch: int = typer.Option(default=1000),
        data_inter: int = typer.Option(default=10000),
        data_total: int = typer.Option(default=1410203),
        file_home: str = typer.Option(default="input/wikimedia"),
        file_name: str = typer.Option(default="wikipedia-20230920-crawl-kowiki.jsonl"),
        table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        table_name: str = typer.Option(default="wikipedia-20230920-parse-kowiki"),
        table_reset: bool = typer.Option(default=True),
        # filter
        filter_min_char: int = typer.Option(default=40),
        filter_min_word: int = typer.Option(default=5),
        filter_black_subtitle: str = typer.Option(default="input/wikimedia/wikipedia-black_sect.txt"),
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
    data_opt = InputOption(
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
            reset=table_reset,
        ) if table_home and table_name else None,
    )
    filter_opt = FilterOption(
        min_char=filter_min_char,
        min_word=filter_min_word,
        black_sect=filter_black_subtitle,
    )
    args = ParseArguments(
        env=env,
        data=data_opt,
        filter=filter_opt,
    )
    tqdm = mute_tqdm_cls()
    save_file = (env.output_home / f"{table_name}-{env.time_stamp}.jsonl")
    assert args.data.file, "data.file is required"
    assert args.data.table, "data.table is required"

    with (
        JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoDBWrapper(args.data.table) as data_table,
        LineFileWrapper(args.data.file) as data_file,
        save_file.open("w") as writer,
    ):
        # parse crawled data
        batches, num_batch, num_input = args.data.load_batches(data_file, args.data.total)
        logger.info(f"Parse from [{args.data.file}] to [{args.data.table}]")
        logger.info(f"- amount: inputs={num_input}, batches={num_batch}")
        logger.info(f"- filter: num_black_sect={args.filter.num_black_sect}, min_char={args.filter.min_char}, min_word={args.filter.min_word}")
        progress, interval = (
            tqdm(batches, total=num_batch, unit="batch", pre="*", desc="parsing"),
            math.ceil(args.data.inter / args.data.batch),
        )
        parsed_ids = set()
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            parse_many(batch=x, wrapper=data_table, parsed_ids=parsed_ids)
        logger.info(progress)

        # save parsed data
        rows, num_row = data_table, len(data_table)
        progress, interval = (
            tqdm(rows, total=num_row, unit="row", pre="*", desc="saving"),
            args.data.inter * 100,
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            writer.write(json.dumps(x, ensure_ascii=False) + '\n')
        logger.info(progress)
        logger.info(f"Saved {num_row} rows to [{save_file}]")


if __name__ == "__main__":
    app()
