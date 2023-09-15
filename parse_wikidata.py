import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from qwikidata.claim import WikidataClaim
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme, ClaimsMixin
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.typedefs import LanguageCode

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments, MongoDBOption
from chrisbase.io import LoggingFormat, pop_keys
from chrisbase.util import to_dataframe, mute_tqdm_cls, wait_future_jobs

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
    from_scratch: bool = field(default=False)
    prog_interval: int = field(default=10000)

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)
        self.lang1_code = LanguageCode(self.lang1)
        self.lang2_code = LanguageCode(self.lang2)


@dataclass
class NetOption(OptionData):
    calling_sec: float = field(default=0.001),
    waiting_sec: float = field(default=300.0),


@dataclass
class ProgramArguments(CommonArguments):
    data: DataOption = field()
    db: MongoDBOption = field()
    net: NetOption | None = field(default=None)
    other: str | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.net, data_prefix="net") if self.net else None,
            to_dataframe(columns=columns, raw=self.db, data_prefix="db") if self.db else None,
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
            to_dataframe(columns=columns, raw={"other": self.other}),
        ]).reset_index(drop=True)


@dataclass
class WikidataUnit(DataClassJsonMixin):
    _id: int
    ns: int
    eid: str
    type: str
    time: str
    lang1: str
    lang2: str
    label1: str | None = None
    label2: str | None = None
    alias1: list = field(default_factory=list)
    alias2: list = field(default_factory=list)
    descr1: str | None = None
    descr2: str | None = None
    # for item only
    title1: str | None = None
    title2: str | None = None
    # for lexeme only
    lang: str | None = None
    cate: str | None = None
    # for all
    claims: list[dict] = field(default_factory=list)


def process_item(i: int, x: dict, args: ProgramArguments):
    with args.db.client() as db:
        table = args.db.table(db)
        if table.count_documents({"_id": i, "eid": x['id']}, limit=1) > 0:
            return

        lang1_code = LanguageCode(args.data.lang1)
        lang2_code = LanguageCode(args.data.lang2)
        row = WikidataUnit(
            _id=i,
            ns=x['ns'],
            eid=x['id'],
            type=x['type'],
            time=x['modified'],
            lang1=args.data.lang1,
            lang2=args.data.lang2,
        )
        if row.type == "item" and row.ns == 0:
            item = WikidataItemEx(x)
            row.label1 = item.get_label(lang1_code)
            row.label2 = item.get_label(lang2_code)
            row.alias1 = item.get_aliases(lang1_code)
            row.alias2 = item.get_aliases(lang2_code)
            row.descr1 = item.get_description(lang1_code)
            row.descr2 = item.get_description(lang2_code)
            row.title1 = item.get_wiki_title(lang1_code)
            row.title2 = item.get_wiki_title(lang2_code)
            row.claims = item.get_truthy_claims()
        elif row.type == "property":
            prop = WikidataPropertyEx(x)
            row.label1 = prop.get_label(lang1_code)
            row.label2 = prop.get_label(lang2_code)
            row.alias1 = prop.get_aliases(lang1_code)
            row.alias2 = prop.get_aliases(lang2_code)
            row.descr1 = prop.get_description(lang1_code)
            row.descr2 = prop.get_description(lang2_code)
            row.claims = prop.get_truthy_claims()
        elif row.type == "lexeme":
            lexm = WikidataLexemeEx(x)
            row.label1 = lexm.get_lemma(lang1_code)
            row.label2 = lexm.get_lemma(lang2_code)
            row.descr1 = lexm.get_gloss(lang1_code)
            row.descr2 = lexm.get_gloss(lang2_code)
            row.lang = lexm.language
            row.cate = lexm.lexical_category
            row.claims = lexm.get_truthy_claims()
        if table.count_documents({"_id": i}, limit=1) > 0:
            table.delete_one({"_id": i})
        table.insert_one(row.to_dict())


def make_jobs(input_list: Iterable[tuple[int, dict]], args: ProgramArguments, pool: ProcessPoolExecutor):
    for i, x in input_list:
        yield i, pool.submit(process_item, i=i, x=x, args=args)


@app.command()
def parse(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="parse_wikidata"),
        output_home: str = typer.Option(default="output-parse_wikidata"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),
        # net
        calling_sec: float = typer.Option(default=0.001),
        waiting_sec: float = typer.Option(default=300.0),
        # data
        input_home: str = typer.Option(default="input/Wikidata"),
        input_name: str = typer.Option(default="latest-all.json.bz2"),
        input_total: int = typer.Option(default=105485440),
        input_limit: int = typer.Option(default=10),
        input_lang1: str = typer.Option(default="ko"),
        input_lang2: str = typer.Option(default="en"),
        from_scratch: bool = typer.Option(default=True),
        prog_interval: int = typer.Option(default=10000),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        output_home=output_home,
        logging_file=logging_file,
        msg_level=logging.DEBUG if debugging else logging.INFO,
        msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
        max_workers=1 if debugging else max(max_workers, 1),
    )
    args = ProgramArguments(
        env=env,
        net=NetOption(
            calling_sec=calling_sec,
            waiting_sec=waiting_sec,
        ),
        db=MongoDBOption(
            tab_name=env.job_name,
            db_name=env.project,
        ),
        data=DataOption(
            home=input_home,
            name=input_name,
            total=input_total,
            lang1=input_lang1,
            lang2=input_lang2,
            limit=input_limit,
            from_scratch=from_scratch,
            prog_interval=prog_interval if prog_interval > 0 else env.max_workers,
        ),
    )
    tqdm = mute_tqdm_cls()
    output_file = (args.env.output_home / f"{args.data.name.stem}.jsonl")

    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        if args.data.from_scratch:
            logger.info(f"Clear database table: {args.db.clear_table()}")
        wikidata_dump = WikidataJsonDump(str(args.data.home / args.data.name))
        input_iter = islice(enumerate(wikidata_dump, start=1), args.data.limit) if args.data.limit > 0 else enumerate(wikidata_dump, start=1)
        input_size = min(args.data.total, args.data.limit) if args.data.limit > 0 else args.data.total
        logger.info(f"Use {args.env.max_workers} workers to parse {input_size} Wikidata items")
        with ProcessPoolExecutor(max_workers=args.env.max_workers) as pool:
            prog_bar = tqdm(make_jobs(input_iter, args, pool), total=input_size, unit="ea", pre="*", desc="importing")
            wait_future_jobs(prog_bar, timeout=args.net.waiting_sec, interval=args.data.prog_interval, pool=pool)
        with output_file.open("w") as out, args.db.client() as db:
            table, find_opt = args.db.table(db), {}
            num_row, rows = table.count_documents(find_opt), table.find(find_opt).sort("_id")
            prog_bar = tqdm(rows, unit="ea", pre="*", desc="exporting", total=num_row)
            for i, row in enumerate(prog_bar, start=1):
                out.write(json.dumps(pop_keys(row, ("claims", "lang1", "lang2")), ensure_ascii=False) + '\n')
                if i % (args.data.prog_interval * 10) == 0:
                    logger.info(prog_bar)
            logger.info(prog_bar)
        logger.info(f"Export {num_row}/{input_size} rows to {output_file}")


if __name__ == "__main__":
    app()
