import json
import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from qwikidata.claim import WikidataClaim
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme, ClaimsMixin
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.typedefs import LanguageCode

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, CommonArguments, OptionData
from chrisbase.data import DataOption, FileOption, TableOption
from chrisbase.data import LineFileWrapper, MongoDBWrapper
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe, mute_tqdm_cls

logger = logging.getLogger(__name__)
app = AppTyper()


class ClaimMixinEx(ClaimsMixin):
    def get_truthy_claims(self) -> list[dict]:
        res = list()
        for claim_group in self.get_truthy_claim_groups().values():
            for claim in claim_group:
                claim: WikidataClaim = claim
                if claim.mainsnak.snaktype == "value" and claim.mainsnak.datavalue is not None:
                    res.append({"property": claim.mainsnak.property_id, "datavalue": claim.mainsnak.datavalue._datavalue_dict})
        return res


class WikidataPropertyEx(WikidataProperty, ClaimMixinEx):
    pass


class WikidataItemEx(WikidataItem, ClaimMixinEx):
    def get_wiki_title(self, lang: LanguageCode) -> str:
        wikiname = f"{lang.lower()}wiki"
        if (
                isinstance(self._entity_dict["sitelinks"], dict)
                and wikiname in self._entity_dict["sitelinks"]
        ):
            return self._entity_dict["sitelinks"][wikiname]["title"]
        else:
            return ""


class WikidataLexemeEx(WikidataLexeme, ClaimMixinEx):
    def get_gloss(self, lang: LanguageCode):
        res = list()
        for sense in self.get_senses():
            res.append({"sense_id": sense.sense_id, "gloss": sense.get_gloss(lang)})
        return res


@dataclass
class WikidataUnit(DataClassJsonMixin):
    _id: str
    ns: int
    type: str
    time: str
    label1: str | None = None
    label2: str | None = None
    title1: str | None = None
    title2: str | None = None
    alias1: list = field(default_factory=list)
    alias2: list = field(default_factory=list)
    descr1: str | None = None
    descr2: str | None = None
    claims: list[dict] = field(default_factory=list)


@dataclass
class FilterOption(OptionData):
    lang1: str = field(default="ko")
    lang2: str = field(default="en")


@dataclass
class ParseArguments(CommonArguments):
    data: DataOption = field()
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
        if not row.label1 or not row.title1:
            return None
        row.alias1 = item.get_aliases(lang1_code)
        row.alias2 = item.get_aliases(lang2_code)
        row.descr1 = item.get_description(lang1_code)
        row.descr2 = item.get_description(lang2_code)
        row.claims = item.get_truthy_claims()
        return row
    elif row.type == "property":
        prop = WikidataPropertyEx(x)
        row.label1 = prop.get_label(lang1_code)
        row.label2 = prop.get_label(lang2_code)
        row.alias1 = prop.get_aliases(lang1_code)
        row.alias2 = prop.get_aliases(lang2_code)
        row.descr1 = prop.get_description(lang1_code)
        row.descr2 = prop.get_description(lang2_code)
        row.claims = prop.get_truthy_claims()
        return row
    elif row.type == "lexeme":
        lexm = WikidataLexemeEx(x)
        row.label1 = lexm.get_lemma(lang1_code)
        row.label2 = lexm.get_lemma(lang2_code)
        if not row.label1:
            return None
        row.descr1 = lexm.get_gloss(lang1_code)
        row.descr2 = lexm.get_gloss(lang2_code)
        row.claims = lexm.get_truthy_claims()
        return row
    return None


def parse_many(batch: Iterable[dict], wrapper: MongoDBWrapper, args: ParseArguments):
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
        debugging: bool = typer.Option(default=False),
        # data
        data_start: int = typer.Option(default=0),
        data_limit: int = typer.Option(default=-1),
        data_batch: int = typer.Option(default=1000),
        data_inter: int = typer.Option(default=10000),
        data_total: int = typer.Option(default=105485440),  # https://www.wikidata.org/wiki/Wikidata:Statistics
        file_home: str = typer.Option(default="input/wikimedia"),
        file_name: str = typer.Option(default="wikidata-20230920-dump.json.bz2"),
        table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        table_name: str = typer.Option(default="wikidata-20230920-parse-kowiki"),
        table_reset: bool = typer.Option(default=True),
        # filter
        filter_lang1: str = typer.Option(default="ko"),
        filter_lang2: str = typer.Option(default="en"),
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
        lang1=filter_lang1,
        lang2=filter_lang2,
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
        JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoDBWrapper(args.data.table) as data_table,
        LineFileWrapper(args.data.file) as data_file,
        save_file.open("w") as writer,
    ):
        # parse dump data
        batches, num_batch, num_input = args.data.input_batches(WikidataJsonDump(str(data_file.path)), args.data.total)
        logger.info(f"Parse from [{args.data.file}] to [{args.data.table}]")
        logger.info(f"- amount: inputs={num_input}, batches={num_batch}")
        logger.info(f"- filter: lang1={args.filter.lang1}, lang2={args.filter.lang2}")
        progress, interval = (
            tqdm(batches, total=num_batch, unit="batch", pre="*", desc="parsing"),
            math.ceil(args.data.inter / args.data.batch),
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            parse_many(batch=x, wrapper=data_table, args=args)
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


@dataclass
class RestoreArguments(CommonArguments):
    data: DataOption = field()

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
        ]).reset_index(drop=True)


@app.command()
def restore(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="restore_wikidata"),
        output_home: str = typer.Option(default="output-restore_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # data
        data_start: int = typer.Option(default=0),
        data_limit: int = typer.Option(default=-1),
        data_batch: int = typer.Option(default=10000),
        data_inter: int = typer.Option(default=100000),
        data_total: int = typer.Option(default=1018174),
        file_home: str = typer.Option(default="input/wikimedia"),
        file_name: str = typer.Option(default="wikidata-20230920-parse-kowiki.jsonl"),
        table_home: str = typer.Option(default="localhost:6382/wikimedia"),
        table_name: str = typer.Option(default="wikidata-20230920-parse-kowiki"),
        table_reset: bool = typer.Option(default=True),
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
    data_opt = DataOption(
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
    args = RestoreArguments(
        env=env,
        data=data_opt,
    )
    tqdm = mute_tqdm_cls()
    save_file = (args.env.output_home / f"{table_name}-{args.env.time_stamp}.jsonl")
    assert args.data.file, "data.file is required"
    assert args.data.table, "data.table is required"

    with (
        JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoDBWrapper(args.data.table) as data_table,
        LineFileWrapper(args.data.file) as data_file,
        save_file.open("w") as writer,
    ):
        # restore parsed data
        batches, num_batch, num_input = args.data.input_batches(data_file, args.data.total)
        logger.info(f"Restore from [{args.data.file}] to [{args.data.table}]")
        logger.info(f"- amount: inputs={num_input}, batches={num_batch}")
        progress, interval = (
            tqdm(batches, total=num_batch, unit="batch", pre="*", desc="restoring"),
            math.ceil(args.data.inter / args.data.batch),
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            data_table.table.insert_many(x)
        logger.info(progress)

        # save restored data
        rows, num_row = data_table, len(data_table)
        progress, interval = (
            tqdm(rows, total=num_row, unit="row", pre="*", desc="saving"),
            args.data.inter,
        )
        for i, x in enumerate(progress):
            if i > 0 and i % interval == 0:
                logger.info(progress)
            writer.write(json.dumps(x, ensure_ascii=False) + '\n')
        logger.info(progress)
        logger.info(f"Saved {num_row} rows to [{save_file}]")


if __name__ == "__main__":
    app()
