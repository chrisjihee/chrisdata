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
from qwikidata.claim import WikidataClaim
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme, ClaimsMixin
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.typedefs import LanguageCode

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments, TableOption, MongoDBTable
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
class DataOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    total: int = field(default=106781030)  # https://www.wikidata.org/wiki/Wikidata:Statistics
    lang1: str = field(default="ko")
    lang2: str = field(default="en")
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
            to_dataframe(columns=columns, raw=self.table, data_prefix="table"),
            to_dataframe(columns=columns, raw={"other": self.other}),
        ]).reset_index(drop=True)


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


def process_one(x: dict, args: ProgramArguments):
    lang1_code = LanguageCode(args.data.lang1)
    lang2_code = LanguageCode(args.data.lang2)
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


def process_many(batch: Iterable[dict], table: Collection, args: ProgramArguments):
    rows = [None if table.count_documents({"_id": x['id']}, limit=1) > 0
            else process_one(x, args) for x in batch]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        table.insert_many(rows)


@app.command()
def parse(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="parse_wikidata"),
        output_home: str = typer.Option(default="output-parse_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        debugging: bool = typer.Option(default=False),
        # data
        input_home: str = typer.Option(default="input/Wikidata"),
        input_name: str = typer.Option(default="latest-all.json.bz2"),
        input_total: int = typer.Option(default=105485440),
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=100),
        input_lang1: str = typer.Option(default="ko"),
        input_lang2: str = typer.Option(default="en"),
        from_scratch: bool = typer.Option(default=False),
        prog_interval: int = typer.Option(default=10000),
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
    args = ProgramArguments(
        env=env,
        data=DataOption(
            home=input_home,
            name=input_name,
            total=input_total,
            lang1=input_lang1,
            lang2=input_lang2,
            limit=input_limit,
            batch=input_batch,
            from_scratch=from_scratch,
            prog_interval=prog_interval,
        ),
        table=TableOption(
            db_host=db_host,
            db_name=env.project,
            tab_name=env.job_name,
        ),
    )
    tqdm = mute_tqdm_cls()
    output_file = (args.env.output_home / f"{args.data.name.stem}-{args.env.time_stamp}.jsonl")

    with (JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='=')):
        with MongoDBTable(args.table) as out_table, output_file.open("w") as out_file:
            if args.data.from_scratch:
                logger.info(f"Clear database table: {args.table}")
                out_table.drop()
            num_input, inputs = args.data.total, WikidataJsonDump(str(args.data.home / args.data.name))
            if args.data.limit > 0:
                num_input, inputs = min(args.data.total, args.data.limit), islice(inputs, args.data.limit)
            num_batch, batches = math.ceil(num_input / args.data.batch), ichunked(inputs, args.data.batch)
            logger.info(f"Parse {num_input} inputs with {num_batch} batches")
            progress, interval = (tqdm(batches, total=num_batch, unit="batch", pre="*", desc="importing"),
                                  math.ceil(args.data.prog_interval / args.data.batch))
            for i, x in enumerate(progress):
                if i > 0 and i % interval == 0:
                    logger.info(progress)
                process_many(batch=x, table=out_table, args=args)
            logger.info(progress)
            find_opt = {}
            num_row, rows = out_table.count_documents(find_opt), out_table.find(find_opt).sort("_id")
            progress, interval = (tqdm(rows, total=num_row, unit="row", pre="*", desc="exporting"),
                                  args.data.prog_interval * 10)
            for i, x in enumerate(progress):
                if i > 0 and i % interval == 0:
                    logger.info(progress)
                out_file.write(json.dumps(x, ensure_ascii=False) + '\n')
            logger.info(progress)
        logger.info(f"Export {num_row}/{num_input} rows to {output_file}")


if __name__ == "__main__":
    app()
