import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from qwikidata.json_dump import WikidataJsonDump

from chrisbase.data import FileStreamer, MongoStreamer
from chrisbase.data import InputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv, CommonArguments, OptionData
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls
from chrisdata.wikidata import *

logger = logging.getLogger(__name__)


@dataclass
class FilterOption(OptionData):
    lang1: str = field(default="ko")
    lang2: str = field(default="en")
    strict: bool = field(default=False)
    truthy: bool = field(default=False)


@dataclass
class ParseArguments(CommonArguments):
    data: InputOption = field()
    filter: FilterOption | None = field(default=None)

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


def parse_one(x: dict, args: ParseArguments):
    if args.env.debugging:
        logger.info('')
        logger.info("*" * 120)
        logger.info(f"parse_one: {x['type']} {x['id']}")
        Path(f"debug-{x['id']}.json").write_text(json.dumps(x, ensure_ascii=False, indent=2))
        logger.info("-" * 120)

    def debug_return(r):
        if args.env.debugging:
            logger.info("*" * 120)
        return r

    lang1_code = LanguageCode(args.filter.lang1)
    lang2_code = LanguageCode(args.filter.lang2)
    row = WikidataUnit(
        _id=x['id'],
        ns=x['ns'],
        type=x['type'],
        time=x['modified'],
    )
    if row.type == "item" and row.ns == 0:
        item = WikidataItemEx(x)
        row.label1 = item.get_label(lang1_code)
        row.label2 = item.get_label(lang2_code)
        row.title1 = item.get_wiki_title(lang1_code)
        row.title2 = item.get_wiki_title(lang2_code)
        if args.filter.strict and (not row.label1 or not row.title1):
            return debug_return(None)
        row.alias1 = item.get_aliases(lang1_code)
        row.alias2 = item.get_aliases(lang2_code)
        row.descr1 = item.get_description(lang1_code)
        row.descr2 = item.get_description(lang2_code)
        row.claims = item.get_claims(args)
        return debug_return(row)
    elif row.type == "property":
        prop = WikidataPropertyEx(x)
        row.label1 = prop.get_label(lang1_code)
        row.label2 = prop.get_label(lang2_code)
        row.alias1 = prop.get_aliases(lang1_code)
        row.alias2 = prop.get_aliases(lang2_code)
        row.descr1 = prop.get_description(lang1_code)
        row.descr2 = prop.get_description(lang2_code)
        row.claims = prop.get_claims(args)
        return debug_return(row)
    elif row.type == "lexeme":
        lexm = WikidataLexemeEx(x)
        row.label1 = lexm.get_lemma(lang1_code)
        row.label2 = lexm.get_lemma(lang2_code)
        if args.filter.strict and not row.label1:
            return debug_return(None)
        row.descr1 = lexm.get_gloss(lang1_code)
        row.descr2 = lexm.get_gloss(lang2_code)
        row.claims = lexm.get_claims(args)
        return debug_return(row)
    return debug_return(None)


def parse_many(batch: Iterable[dict], wrapper: MongoStreamer, args: ParseArguments):
    rows = [parse_one(x, args) if wrapper.count({"_id": x['id']}) == 0 else None
            for x in batch]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        wrapper.table.insert_many(rows)


@app.command()
def parse(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="parse_wikidata"),
        output_home: str = typer.Option(default="output-parse_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=True),
        # data
        data_start: int = typer.Option(default=0),
        data_limit: int = typer.Option(default=-1),
        data_batch: int = typer.Option(default=1000),
        data_inter: int = typer.Option(default=10000),
        data_total: int = typer.Option(default=105485440),  # https://www.wikidata.org/wiki/Wikidata:Statistics
        file_home: str = typer.Option(default="input/Wikidata"),
        file_name: str = typer.Option(default="wikidata-20230911-all.json.bz2"),
        table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        table_name: str = typer.Option(default="wikidata-20230911-all-parse-ko-en"),
        table_reset: bool = typer.Option(default=True),
        # filter
        filter_lang1: str = typer.Option(default="ko"),
        filter_lang2: str = typer.Option(default="en"),
        filter_strict: bool = typer.Option(default=False),
        filter_truthy: bool = typer.Option(default=False),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        output_home=output_home,
        logging_file=logging_file,
        msg_level=logging.INFO,  # if not debugging else logging.DEBUG,
        msg_format=LoggingFormat.CHECK_24,  # if not debugging else LoggingFormat.DEBUG_24,
    )
    data_opt = InputOption(
        start=data_start,
        limit=data_limit if not debugging else 1,
        batch=data_batch if not debugging else 2,
        inter=data_inter if not debugging else 1,
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
        lang1=filter_lang1,
        lang2=filter_lang2,
        strict=filter_strict,
        truthy=filter_truthy,
    )
    args = ParseArguments(
        env=env,
        data=data_opt,
        filter=filter_opt,
    )
    tqdm = mute_tqdm_cls()
    save_file = (args.env.output_home / f"{table_name}-{args.env.time_stamp}.jsonl")
    assert args.data.file, "data.file is required"
    assert args.data.table, "data.table is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoStreamer(args.data.table) as data_table,
        FileStreamer(args.data.file) as data_file,
        save_file.open("w") as writer,
    ):
        # parse dump data
        inputs = args.data.ready_inputs(WikidataJsonDump(str(data_file.path)), data_total)
        logger.info(f"Parse from [{data_file.opt}] to [{data_table.opt}]")
        logger.info(f"- amount: inputs={data_total}, batches={inputs.total}")
        logger.info(f"- filter: lang1={args.filter.lang1}, lang2={args.filter.lang2}")
        progress, interval = (
            tqdm(inputs.batches, total=inputs.total, unit="batch", pre="*", desc="parsing"),
            math.ceil(args.data.inter / args.data.batch),
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            parse_many(batch=x, wrapper=data_table, args=args)
        logger.info(progress)

        # save parsed data
        progress, interval = (
            tqdm(data_table, total=len(data_table), unit="row", pre="*", desc="saving"),
            args.data.inter * 100,
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            writer.write(json.dumps(x, ensure_ascii=False, indent=None if not debugging else 2) + '\n')
        logger.info(progress)
        logger.info(f"Saved {len(data_table)} rows to [{save_file}]")
