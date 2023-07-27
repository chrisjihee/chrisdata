import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

import httpx
import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments
from chrisbase.io import LoggingFormat
from chrisbase.util import MongoDB, to_dataframe, time_tqdm_cls, mute_tqdm_cls, wait_future_jobs, LF
from wikipediaapi import Wikipedia
from wikipediaapi import WikipediaPage

logger = logging.getLogger(__name__)
app = AppTyper()
mongos: List[MongoDB] = []


class WikipediaEx(Wikipedia):
    def __init__(self, *args, ip=None, max_retrial=10, retrial_sec=10.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if ip:
            self._session.close()
            self._session = httpx.Client(transport=httpx.HTTPTransport(local_address=ip))
        self.max_retrial = max_retrial
        self.retrial_sec = retrial_sec

    def _query(self, page: WikipediaPage, params: Dict[str, Any]):
        for n in range(self.max_retrial):
            try:
                return super()._query(page, params)
            except httpx.HTTPError as e:
                print()
                logger.warning(f"{type(e).__qualname__} on _query(): {e} || n={n} || title={page.title}")
                time.sleep(self.retrial_sec)
        else:
            raise RuntimeError(f"Failed to query {page.title} after {self.max_retrial} retrials")


apis: List[WikipediaEx] = []


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
    calling_sec: float = field(default=0.5)
    waiting_sec: float = field(default=300.0),
    retrial_sec: float = field(default=10.0),
    max_retrial: int = field(default=10),


@dataclass
class DataOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    lang: str | Path = field()
    limit: int = field(default=-1)

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)


@dataclass
class ProgramArguments(CommonArguments):
    net: NetOption | None = field(default=None)
    data: DataOption | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.net, data_prefix="net"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
        ]).reset_index(drop=True)


def load_query_list(args: ProgramArguments, limit: int | None = None) -> List[str]:
    assert (args.data.home / args.data.name).exists(), f"No input file: {args.data.home / args.data.name}"
    global apis
    for ip in args.env.ip_addrs:
        apis.append(WikipediaEx(user_agent=f"{args.env.project}/1.0", language=args.data.lang, ip=ip,
                                max_retrial=args.net.max_retrial, retrial_sec=args.net.retrial_sec,
                                timeout=args.net.waiting_sec))
    with open(args.data.home / args.data.name) as f:
        if not limit or limit < 1:
            return f.read().splitlines()
        else:
            return f.read().splitlines()[:limit]


@dataclass
class ProcessResult(DataClassJsonMixin):
    _id: int
    query: str
    title: str | None = None
    page_id: int | None = None
    section_list: list = field(default_factory=list)
    passage_list: list = field(default_factory=list)


def process_query(i: int, x: str, s: float | None = None):
    if s and s > 0:
        time.sleep(s)
    api = apis[i % len(apis)]
    page: WikipediaPage = api.page(x)
    if not page.exists():
        result = ProcessResult(_id=i, query=x)
    else:
        result = ProcessResult(_id=i, query=x, title=page.title, page_id=page.pageid)
        result.section_list.append((x, '', '', page.summary))
        result.section_list += get_section_list_lv2(x, page.sections)
        result.passage_list = get_passage_list(result.section_list, page.pageid)
    for mongo in mongos:
        mongo.table.insert_one(result.to_dict())


@app.command()
def crawl(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="crawl_wikipedia"),
        debugging: bool = typer.Option(default=False),
        max_workers: int = typer.Option(default=os.cpu_count()),
        output_home: str = typer.Option(default="output-crawl_wikipedia"),
        # net
        calling_sec: float = typer.Option(default=0.5),
        waiting_sec: float = typer.Option(default=300.0),
        retrial_sec: float = typer.Option(default=10.0),
        max_retrial: int = typer.Option(default=10),
        # data
        input_home: str = typer.Option(default="input"),
        input_name: str = typer.Option(default="kowiki-sample.txt"),
        input_lang: str = typer.Option(default="ko"),
        input_limit: int = typer.Option(default=-1),
        # etc
        use_tqdm: bool = typer.Option(default=True),
):
    args = ProgramArguments(
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
            calling_sec=calling_sec,
            waiting_sec=waiting_sec,
            retrial_sec=retrial_sec,
            max_retrial=max_retrial,
        ),
        data=DataOption(
            home=input_home,
            name=input_name,
            lang=input_lang,
            limit=input_limit,
        ),
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("wikipediaapi").setLevel(logging.WARNING)
    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        with MongoDB(db_name=args.env.project, tab_name=args.env.job_name, clear_table=True, pool=mongos) as mongo:
            failed_ids: List[int] = []
            query_list = load_query_list(args=args, limit=args.data.limit)
            logger.info(f"Use {args.env.max_workers} workers to crawl {len(query_list)} wikipedia queries")
            job_tqdm = time_tqdm_cls(bar_size=100, desc_size=9) if use_tqdm else mute_tqdm_cls()
            if args.env.max_workers < 2:
                for i, x in enumerate(job_tqdm(query_list, pre="┇", desc="visiting", unit="job")):
                    process_query(i=i + 1, x=x)
            else:
                with ProcessPoolExecutor(max_workers=args.env.max_workers) as pool:
                    jobs: Dict[int, Future] = {}
                    for i, x in enumerate(query_list):
                        jobs[i] = pool.submit(process_query, i=i + 1, x=x, s=args.net.calling_sec)
                    failed_ids = wait_future_jobs(job_tqdm(jobs.items(), pre="┇", desc="visiting", unit="job"),
                                                  timeout=args.net.waiting_sec, pool=pool)
            logger.info(f"Success: {mongo.num_documents}/{len(query_list)}")
            logger.info(f"Failure: {len(failed_ids)}/{len(query_list)}")
            mongo.output_table(to=args.env.output_home / f"{args.env.job_name}.jsonl")
            if failed_ids:
                logger.info(f"Failed IDs: {failed_ids}")
                failed_query_list = [query_list[i] for i in failed_ids]
                failed_query_file = args.env.output_home / f"{args.data.name.stem}-failed{args.data.name.suffix}"
                failed_query_file.write_text("\n".join(failed_query_list))


if __name__ == "__main__":
    app()
