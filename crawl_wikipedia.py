import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin

from chrisbase.data import AppTyper, ProjectEnv, OptionData, CommonArguments, JobTimer
from chrisbase.io import LoggingFormat
from chrisbase.net import ips
from chrisbase.util import to_dataframe, MongoDB
from wikipediaapi import Wikipedia
from wikipediaapi import WikipediaPage

logger = logging.getLogger(__name__)
app = AppTyper()


class WikipediaFromIP(Wikipedia):
    def __init__(self, *args, ip=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if ip:
            self._session.close()
            self._session = httpx.Client(transport=httpx.HTTPTransport(local_address=ip))

    def _query(self, *args, **kwargs):
        return super()._query(*args, **kwargs)

    # def _query_2(self, *args, **kwargs):
    #     logger.info("-" * 160)
    #     r = super()._query(*args, **kwargs)
    #     logger.info(f"session type={type(self._session)}")
    #     if isinstance(self._session, httpx.Client):
    #         logger.info("local_address=%s", self._session._transport._pool.connections[0]._local_address)
    #     return r


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
class DataOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    lang: str | Path = field()

    def __post_init__(self):
        self.home = Path(self.home)


@dataclass
class WikiCrawlArguments(CommonArguments):
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
    _id: str
    query: str
    title: str
    section_list: list = field(default_factory=list)
    passage_list: list = field(default_factory=list)


apis = []


def query_to_result(api_idx, query) -> WikiCrawlResult | None:
    api = apis[api_idx % len(apis)]
    page: WikipediaPage = api.page(query)
    if page.exists() and len(page.summary.strip()) > 0:
        res = WikiCrawlResult(_id=' || '.join([str(page.pageid), query]), query=query, title=page.title)
        res.section_list.append((query, '', '', page.summary))
        res.section_list += get_section_list_lv2(query, page.sections)
        if len(res.section_list) > 1:
            res.passage_list = get_passage_list(res.section_list, page.pageid)
            return res
    return None


@app.command()
def crawl(
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="crawl_wikipedia"),
        debugging: bool = typer.Option(default=False),
        max_workers: int = typer.Option(default=os.cpu_count()),
        output_home: str = typer.Option(default="output"),
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
        data=DataOption(
            home=input_home,
            name=input_name,
            lang=input_lang,
        ),
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        assert (args.data.home / args.data.name).exists(), f"No input file: {args.data.home / args.data.name}"
        with open(args.data.home / args.data.name) as f:
            input_queries = f.read().splitlines()

        logger.info(f"Use {args.env.max_workers} workers to crawl {len(input_queries)} wikipedia pages")
        global apis
        for ip in ips:
            apis.append(WikipediaFromIP(user_agent=f"{args.env.project}/1.0", language=args.data.lang, ip=ip))

        with MongoDB(db_name=args.env.project, tab_name=args.env.job_name) as mongo:
            if args.env.max_workers < 2:
                for i, query in enumerate(input_queries):
                    res: WikiCrawlResult | None = query_to_result(api_idx=i, query=query)
                    if mongo and res:
                        mongo.table.insert_one(res.to_dict())
            else:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                pool = ProcessPoolExecutor(max_workers=args.env.max_workers)
                jobs = [pool.submit(query_to_result, api_idx=i, query=query) for i, query in enumerate(input_queries)]
                for job in as_completed(jobs):
                    res: WikiCrawlResult | None = job.result()
                    if mongo and res:
                        mongo.table.insert_one(res.to_dict())
            mongo.output_table(to=args.env.output_home / f"{args.env.job_name}.jsonl", include_id=True)
            # for row in mongo.table.find():
            #     res = WikiCrawlResult.from_dict(row)
            #     logger.info(res.to_json(ensure_ascii=False, indent=4))


if __name__ == "__main__":
    app()
