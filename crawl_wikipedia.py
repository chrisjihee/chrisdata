import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

import httpx
import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin

from chrisbase.data import AppTyper, ProjectEnv, OptionData, CommonArguments, JobTimer
from chrisbase.io import LoggingFormat
from chrisbase.net import ips
from chrisbase.proc import all_future_results
from chrisbase.util import to_dataframe, MongoDB
from wikipediaapi import Wikipedia
from wikipediaapi import WikipediaPage

logger = logging.getLogger(__name__)
app = AppTyper()
savers: List[MongoDB] = []


class WikipediaFromIP(Wikipedia):
    def __init__(self, *args, ip=None, max_retrial=3, retrial_sec=3.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if ip:
            self._session.close()
            self._session = httpx.Client(transport=httpx.HTTPTransport(local_address=ip))
        self.max_retrial = max_retrial
        self.retrial_sec = retrial_sec

    def _query(self, page: "WikipediaPage", params: Dict[str, Any]):
        for i in range(self.max_retrial):
            try:
                return super()._query(page, params)
            except httpx.HTTPError as e:
                logger.warning(f"[{type(e)}] on _query(): (i={i}) (e.args={e.args}) {e}")
                time.sleep(self.retrial_sec)
        else:
            raise RuntimeError(f"Failed to query {page} after {self.max_retrial} retrials")


apis: List[WikipediaFromIP] = []


def get_passage_list(section_list, page_id):
    passage_list = []
    for i, s in enumerate(section_list):
        passage = {}

        passage["파일명"] = s[0]
        passage["문서제목"] = s[0]

        passage["문서번호"] = str(page_id)
        passage["장번호"] = str(i)
        passage["절번호"] = '0'
        passage["조번호"] = '0'
        passage["단위번호"] = '%s-%s-%s-%s' % (passage["문서번호"], passage["장번호"], passage["절번호"], passage["조번호"])

        passage["장제목"] = s[1]
        passage["절제목"] = s[2]
        passage["조제목"] = ''

        passage["조내용"] = ''
        sent_list = [s[0] + '. ']
        if s[1] != '':    sent_list.append(s[1] + '. ')
        if s[2] != '':    sent_list.append(s[2] + '. ')
        for text in s[3].split('\n'):
            if text.strip() != '':
                sent_list.append(text.strip() + '\n')

        for sent_i, text in enumerate(sent_list):
            passage["조내용"] += text
            passage["문장%d" % (sent_i)] = text[:-1]
        passage["조내용"] = passage["조내용"][:-1]

        passage["html"] = ''
        passage["소스"] = 'passage'
        passage["타입"] = 'passage'

        passage_list.append(passage)

    return passage_list


def get_subsections(sections):
    if not sections:
        return ''
    sub_s = ''
    for s in sections:
        sub_s += get_subsections(s.sections)
    return s.title + '\n' + s.text + '\n' + sub_s


def get_section_list_lv2(title, sections):
    section_list = []
    for s1 in sections:
        if s1.text != '':
            section_list.append((title, s1.title, '', s1.text))

        if s1.sections:
            for s2 in s1.sections:
                text = s2.text
                if s2.sections:
                    sub_s = get_subsections(s2.sections)
                    text += '\n' + sub_s
                section_list.append((title, s1.title, s2.title, text))
    return section_list


@dataclass
class NetOption(OptionData):
    max_retrial: int = field(default=3),
    waiting_sec: float = field(default=30.0),
    retrial_sec: float = field(default=3.0),
    concurrent_sec: float = field(default=0.3)


@dataclass
class DataOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    lang: str | Path = field()

    def __post_init__(self):
        self.home = Path(self.home)


@dataclass
class WikiCrawlArguments(CommonArguments):
    net: NetOption | None = field(default=None)
    data: DataOption | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
        ]).reset_index(drop=True)


@dataclass
class WikiCrawlResult(DataClassJsonMixin):
    _id: int
    query: str
    title: str | None = None
    page_id: int | None = None
    section_list: list = field(default_factory=list)
    passage_list: list = field(default_factory=list)


def process_query(i: int, x: str, s: float | None = None) -> int:
    api = apis[i % len(apis)]
    if s and s > 0.0:
        time.sleep(s)
    page: WikipediaPage = api.page(x)
    if not page.exists():
        result = WikiCrawlResult(_id=i, query=x)
    else:
        result = WikiCrawlResult(_id=i, query=x, title=page.title, page_id=page.pageid)
        result.section_list.append((x, '', '', page.summary))
        result.section_list += get_section_list_lv2(x, page.sections)
        result.passage_list = get_passage_list(result.section_list, page.pageid)
    for saver in savers:
        saver.table.insert_one(result.to_dict())
    return 1 if page.exists() else 0


@app.command()
def crawl(
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="crawl_wikipedia"),
        debugging: bool = typer.Option(default=True),
        max_workers: int = typer.Option(default=os.cpu_count()),
        max_retrial: int = typer.Option(default=3),
        retrial_sec: float = typer.Option(default=3.0),
        waiting_sec: float = typer.Option(default=30.0),
        concurrent_sec: float = typer.Option(default=0.3),
        output_home: str = typer.Option(default="output-crawl_wikipedia"),
        input_home: str = typer.Option(default="input"),
        input_name: str = typer.Option(default="kowiki-sample.txt"),
        input_lang: str = typer.Option(default="ko"),
):
    args = WikiCrawlArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name,
            debugging=debugging,
            output_home=output_home,
            msg_level=logging.DEBUG if debugging else logging.INFO,
            msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_24,
            max_workers=1 if debugging else max(max_workers, 1),
        ),
        net=NetOption(
            max_retrial=max_retrial,
            retrial_sec=retrial_sec,
            waiting_sec=waiting_sec,
            concurrent_sec=concurrent_sec,
        ),
        data=DataOption(
            home=input_home,
            name=input_name,
            lang=input_lang,
        ),
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("wikipediaapi").setLevel(logging.WARNING)
    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        assert (args.data.home / args.data.name).exists(), f"No input file: {args.data.home / args.data.name}"
        with open(args.data.home / args.data.name) as f:
            input_queries = f.read().splitlines()[:50000]  # TODO: temporary slicing

        global apis
        for ip in ips:
            apis.append(WikipediaFromIP(user_agent=f"{args.env.project}/1.0", language=args.data.lang, ip=ip,
                                        max_retrial=args.net.max_retrial, retrial_sec=args.net.retrial_sec,
                                        timeout=args.net.waiting_sec))

        with MongoDB(db_name=args.env.project, tab_name=args.env.job_name, clear_table=True, pool=savers) as mongo:
            logger.info(f"Use {args.env.max_workers} workers to crawl {len(input_queries)} wikipedia pages")
            if args.env.max_workers < 2:
                num_success = sum(process_query(i=i + 1, x=x) for i, x in enumerate(input_queries))
            else:
                pool = ProcessPoolExecutor(max_workers=args.env.max_workers)
                jobs = [pool.submit(process_query, i=i + 1, x=x, s=args.net.concurrent_sec) for i, x in enumerate(input_queries)]
                num_success = sum(all_future_results(pool, jobs, default=0, timeout=args.net.waiting_sec))
            logger.info(f"Success: {num_success}/{len(input_queries)}")
            mongo.output_table(to=args.env.output_home / f"{args.env.job_name}.jsonl")


if __name__ == "__main__":
    app()
