import json
import math
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin

from chrisbase.data import InputOption, OutputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv, CommonArguments, OptionData
from chrisbase.data import Streamer, FileStreamer, MongoStreamer
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls
from . import *

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
            to_dataframe(columns=columns, raw=self.output.file, data_prefix="output.file") if self.output.file else None,
            to_dataframe(columns=columns, raw=self.output.table, data_prefix="output.table") if self.output.table else None,
            to_dataframe(columns=columns, raw=self.filter, data_prefix="filter"),
        ]).reset_index(drop=True)


@dataclass
class PassageUnit(DataClassJsonMixin):
    _id: str
    title: str
    subtitle1: str
    subtitle2: str
    body_text: str


def parse_one(x: dict, parsed_ids: set[int], opt: FilterOption) -> Iterable[PassageUnit]:
    doc: WikipediaCrawlResult = WikipediaCrawlResult.from_dict(x)
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


def parse_many(batch: Iterable[dict], wrapper: MongoStreamer, parsed_ids: set[int], filter_opt: FilterOption):
    batch_units = [x for x in [parse_one(x, parsed_ids, opt=filter_opt) for x in batch] if x]
    all_units = [unit for batch in batch_units for unit in batch]
    rows = [row.to_dict() for row in all_units if row]
    if len(rows) > 0:
        wrapper.table.insert_many(rows)


@app.command()
def parse(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="parse_wikipedia"),
        output_home: str = typer.Option(default="output-parse_wikipedia"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=1000),
        input_inter: int = typer.Option(default=10000),
        input_total: int = typer.Option(default=1410203),
        input_file_home: str = typer.Option(default="input/Wikipedia"),
        input_file_name: str = typer.Option(default="kowiki-20230701-all-titles-in-ns0.jsonl"),
        # output
        output_file_home: str = typer.Option(default="input/Wikipedia"),
        output_file_name: str = typer.Option(default="kowiki-20230701-all-titles-in-ns0-parse.jsonl"),
        output_table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        output_table_name: str = typer.Option(default="kowiki-20230701-all-titles-in-ns0-parse"),
        output_table_reset: bool = typer.Option(default=True),
        # filter
        filter_min_char: int = typer.Option(default=40),
        filter_min_word: int = typer.Option(default=5),
        filter_black_sect: str = typer.Option(default="input/Wikipedia/wikipedia-black_sect.txt"),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        logging_home=output_home,
        logging_file=logging_file,
        message_level=logging.DEBUG if debugging else logging.INFO,
        message_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
    )
    input_opt = InputOption(
        start=input_start,
        limit=input_limit,
        batch=input_batch,
        inter=input_inter,
        file=FileOption(
            home=input_file_home,
            name=input_file_name,
        ) if input_file_home and input_file_name else None,
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
        min_word=filter_min_word,
        black_sect=filter_black_sect,
    )
    args = ParseArguments(
        env=env,
        input=input_opt,
        output=output_opt,
        filter=filter_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.file, "input.file is required"
    assert args.output.file, "output.file is required"
    assert args.output.table, "output.table is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoStreamer(args.output.table) as output_table,
        FileStreamer(args.output.file) as output_file,
        FileStreamer(args.input.file) as input_file,
    ):
        # parse crawled data
        reader = Streamer.first_usable(input_file)
        writer = Streamer.first_usable(output_table)
        inputs = args.input.ready_inputs(reader, input_total)
        logger.info(f"Parse from [{reader.opt}] to [{writer.opt}]")
        logger.info(f"- amount: inputs={input_total}, batches={args.input.batch}")
        logger.info(f"- filter: num_black_sect={args.filter.num_black_sect}, min_char={args.filter.min_char}, min_word={args.filter.min_word}")
        progress, interval = (
            tqdm(inputs.items, total=inputs.total, unit="batch", pre="*", desc="parsing"),
            math.ceil(args.input.inter / args.input.batch),
        )
        parsed_ids = set()
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            parse_many(batch=x, wrapper=output_table, parsed_ids=parsed_ids, filter_opt=args.filter)
        logger.info(progress)

        # save parsed data
        progress, interval = (
            tqdm(writer, total=len(writer), unit="row", pre="*", desc="saving"),
            args.input.inter * 100,
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            output_file.fp.write(json.dumps(x, ensure_ascii=False) + '\n')
        logger.info(progress)
        logger.info(f"Saved {len(writer)} rows to [{output_file.path}]")


if __name__ == "__main__":
    app()
