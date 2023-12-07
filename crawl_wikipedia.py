import json
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import List, Dict, Any, Tuple

import httpx
import pandas as pd
import typer
from dataclasses_json import DataClassJsonMixin
from wikipediaapi import Wikipedia
from wikipediaapi import WikipediaPage

from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments
from chrisbase.io import LoggingFormat
from chrisbase.util import MongoDB, to_dataframe, mute_tqdm_cls, wait_future_jobs, LF

mongos: List[MongoDB] = []
logger = logging.getLogger(__name__)
app = AppTyper()


class WikipediaEx(Wikipedia):
    def __init__(self, *args, ip=None, max_retrial=10, retrial_sec=10.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if ip:
            self._httpx_client = httpx.Client(
                transport=httpx.HTTPTransport(local_address=ip),
                timeout=httpx.Timeout(timeout=120.0)
            )
        self.max_retrial = max_retrial
        self.retrial_sec = retrial_sec

    def __del__(self) -> None:
        super().__del__()
        try:
            self._httpx_client.close()
        except Exception:
            pass

    def _query(self, page: WikipediaPage, params: Dict[str, Any]):
        for n in range(self.max_retrial):
            try:
                base_url = "https://" + page.language + ".wikipedia.org/w/api.php"
                logger.debug(
                    "Request URL: %s",
                    base_url + "?" + "&".join([k + "=" + str(v) for k, v in params.items()]),
                )
                params["format"] = "json"
                params["redirects"] = 1
                r = self._httpx_client.get(base_url, params=params, **self._request_kwargs)
                return r.json()
            except httpx.HTTPError as e:
                logger.warning(f"{type(e).__qualname__} on _query(): {e} || n={n} || title={page.title}")
                time.sleep(self.retrial_sec)
        else:
            raise RuntimeError(f"Failed to query {page.title} after {self.max_retrial} retrials")


api_list_per_ip: List[WikipediaEx] = []


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
    lang: str = field(default="ko")
    limit: int = field(default=-1)
    from_scratch: bool = field(default=False)
    prog_interval: int = field(default=1)

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)


@dataclass
class NetOption(OptionData):
    calling_sec: float = field(default=0.001),
    waiting_sec: float = field(default=300.0),
    retrial_sec: float = field(default=3.0),
    max_retrial: int = field(default=10),


@dataclass
class ProgramArguments(CommonArguments):
    data: DataOption = field()
    net: NetOption | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.net, data_prefix="net") if self.net else None,
            to_dataframe(columns=columns, raw=self.data, data_prefix="data"),
        ]).reset_index(drop=True)


def load_query_list(args: ProgramArguments) -> List[Tuple[int, str]]:
    assert (args.data.home / args.data.name).exists(), f"No input file: {args.data.home / args.data.name}"
    with open(args.data.home / args.data.name) as f:
        lines = f.read().splitlines()
        lines = islice(lines, args.data.limit) if args.data.limit > 0 else lines
    rows = [x.split("\t") for x in lines]
    if len(rows[0]) < 2:
        return [(i, row[0]) for i, row in enumerate(rows, start=1)]
    else:
        return [(int(row[0]), row[1]) for row in rows]


def reset_global_api(args: ProgramArguments):
    assert args.net, "No net option on args"
    assert args.data.lang, "No lang option on args.data"
    global api_list_per_ip
    api_list_per_ip.clear()
    for ip in args.env.ip_addrs:
        api_list_per_ip.append(WikipediaEx(user_agent=f"{args.env.project}/1.0", language=args.data.lang, ip=ip,
                                           max_retrial=args.net.max_retrial, retrial_sec=args.net.retrial_sec,
                                           timeout=args.net.waiting_sec))
    return len(api_list_per_ip)


@dataclass
class ProcessResult(DataClassJsonMixin):
    _id: int
    query: str
    title: str | None = None
    page_id: int | None = None
    section_list: list = field(default_factory=list)
    passage_list: list = field(default_factory=list)


def process_query(i: int, x: str, s: float | None = None):
    is_done = all(mongo.table.count_documents({"_id": i, "query": x}, limit=1) > 0 for mongo in mongos)
    if is_done:
        return
    if s and s > 0:
        time.sleep(s)
    api = api_list_per_ip[i % len(api_list_per_ip)]
    page: WikipediaPage = api.page(x)
    result = ProcessResult(_id=i, query=x)
    page_exists = False
    try:
        page_exists = page.exists()
    except KeyError:
        logger.warning(f"KeyError on process_query(i={i}, x={x})")
    if page_exists:
        result.title = page.title
        result.page_id = page.pageid
        result.section_list.append((x, '', '', page.summary))
        result.section_list += get_section_list_lv2(x, page.sections)
        result.passage_list = get_passage_list(result.section_list, page.pageid)
    for mongo in mongos:
        if mongo.table.count_documents({"_id": i}, limit=1) > 0:
            mongo.table.delete_one({"_id": i})
        mongo.table.insert_one(result.to_dict())


def table_name(args: ProgramArguments) -> str:
    postfix = re.sub("-failed.*", "", args.data.name.stem)
    return f"{args.env.job_name}-{postfix}"


@app.command()
def crawl(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="crawl_wikipedia"),
        output_home: str = typer.Option(default="output-crawl_wikipedia"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=10),
        debugging: bool = typer.Option(default=False),
        # net
        calling_sec: float = typer.Option(default=0.001),
        waiting_sec: float = typer.Option(default=300.0),
        retrial_sec: float = typer.Option(default=3.0),
        max_retrial: int = typer.Option(default=10),
        # data
        input_home: str = typer.Option(default="input"),
        input_name: str = typer.Option(default="kowiki-sample.txt"),
        input_lang: str = typer.Option(default="ko"),
        input_limit: int = typer.Option(default=100),
        from_scratch: bool = typer.Option(default=False),
        prog_interval: int = typer.Option(default=15),
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
            retrial_sec=retrial_sec,
            max_retrial=max_retrial,
        ),
        data=DataOption(
            home=input_home,
            name=input_name,
            lang=input_lang,
            limit=input_limit,
            from_scratch=from_scratch,
            prog_interval=prog_interval if prog_interval > 0 else env.max_workers,
        ),
    )
    tqdm = mute_tqdm_cls()
    output_file = (args.env.output_home / f"{args.data.name.stem}.jsonl")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("wikipediaapi").setLevel(logging.WARNING)
    with JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        with MongoDB(db_name=args.env.project, tab_name=table_name(args), clear_table=args.data.from_scratch, pool=mongos, port=6382) as mongo:
            input_list = load_query_list(args=args)
            input_size = len(input_list)
            num_global_api = reset_global_api(args=args)
            logger.info(f"Use {num_global_api} apis and {args.env.max_workers} workers to crawl {input_size} wikipedia queries")
            with ProcessPoolExecutor(max_workers=args.env.max_workers) as pool:
                jobs = [(i, pool.submit(process_query, i=i, x=x, s=args.net.calling_sec)) for i, x in input_list]
                prog_bar = tqdm(jobs, unit="ea", pre="*", desc="visiting")
                wait_future_jobs(prog_bar, timeout=args.net.waiting_sec, interval=args.data.prog_interval, pool=pool)
            done_ids = set()
            with output_file.open("w") as out:
                row_filter = {}
                num_row, rows = mongo.table.count_documents(row_filter), mongo.table.find(row_filter).sort("_id")
                prog_bar = tqdm(rows, unit="ea", pre="*", desc="exporting", total=num_row)
                for i, row in enumerate(prog_bar, start=1):
                    done_ids.add(row.get("_id"))
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    if i % (args.data.prog_interval * 10) == 0:
                        logger.info(prog_bar)
                logger.info(prog_bar)
            logger.info(f"Export {num_row}/{input_size} rows to {output_file}")
            undone_inputs = [(i, x) for i, x in input_list if i not in done_ids]
            if undone_inputs:
                logger.info(f"Found {len(undone_inputs)} undone inputs")
                undone_file = args.data.home / f"{args.data.name.stem}-failed.tsv"
                undone_file.write_text(LF.join([f"{i}\t{x}" for i, x in undone_inputs]))


if __name__ == "__main__":
    app()
