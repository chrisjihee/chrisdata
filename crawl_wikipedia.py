import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import typer

import wikipediaapi
from chrisbase.data import AppTyper, ArgumentsUsing
from chrisbase.data import ProjectEnv, OptionData, CommonArguments
from chrisbase.io import JobTimer
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe

logger = logging.getLogger(__name__)
app = AppTyper()


def get_json(section_list, doc_index):
    passage_list = []
    for i, s in enumerate(section_list):
        passage = {}

        passage["파일명"] = s[0]
        passage["문서제목"] = s[0]

        passage["문서번호"] = str(doc_index)
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
    tag = None
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


@app.command()
def crawl(
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=True),
        output_home: str = typer.Option(default="output"),
        input_home: str = typer.Option(default="input"),
        input_name: str = typer.Option(default="sample.txt"),
        input_lang: str = typer.Option(default="ko"),
):
    args = WikiCrawlArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else f"WikiCrawl-from-{input_name}",
            output_home=output_home,
            debugging=debugging,
            msg_level=logging.DEBUG if debugging else logging.INFO,
            msg_format=LoggingFormat.DEBUG_36 if debugging else LoggingFormat.CHECK_24,
        ),
        data=DataOption(
            home=input_home,
            name=input_name,
            lang=input_lang,
        ),
    )
    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        with ArgumentsUsing(args.info_arguments(), delete_on_exit=False):
            api = wikipediaapi.Wikipedia(f"{args.env.project}/1.0", args.data.lang)
            with open(args.data.home / args.data.name) as f:
                titles = f.read().splitlines()
            logger.info(f"Let's extract from {len(titles)} wikipedia pages!")


if __name__ == "__main__":
    app()

    # raw_outpath = pathlib.Path(args.outfolder1)
    # json_outpath = pathlib.Path(args.outfolder2)
    #
    # fileindex = 0
    # filename = str(fileindex).zfill(5)
    #
    # outfile_raw = raw_outpath / filename
    # f_raw = open(outfile_raw, 'w', encoding='utf-8')
    # outfile_json = json_outpath / (filename + '.json')
    # f_json = open(outfile_json, "w", encoding='utf-8')
    #
    # for index, title in enumerate(titles):
    #     if title.find(':') != -1:
    #         continue
    #
    #     if index % 1000 == 0 and index != 0:
    #         f_raw.close()
    #         f_json.close()
    #
    #         fileindex += 1
    #         filename = str(fileindex).zfill(5)
    #
    #         outfile_raw = raw_outpath / filename
    #         f_raw = open(outfile_raw, 'w', encoding='utf-8')
    #         outfile_json = json_outpath / (filename + '.json')
    #         f_json = open(outfile_json, "w", encoding='utf-8')
    #
    #     page = wiki.page(title)
    #     from sys import stderr
    #
    #     print(f'{page.exists()}\t{title}\t{page.title}\t{filename}', file=stderr)
    #
    #     try:
    #         if page.exists():
    #             section_list = []
    #             section_list.append((title, '', '', page.summary))
    #             section_list += get_section_list_lv2(title, page.sections)
    #
    #             if page.summary == '' and len(section_list) == 1:
    #                 continue
    #
    #             json.dump(section_list, f_raw, ensure_ascii=False, indent=4)
    #             f_raw.write('\n\n')
    #
    #             passage_json = get_json(section_list, index)
    #             for passage in passage_json:
    #                 json.dump(passage, f_json, ensure_ascii=False, indent=4, sort_keys=True)
    #                 f_json.write('\n\n')
    #
    #     except Exception as ex:
    #         print(title)
    #         print('[Error]\t', ex)
    #
    # f_raw.close()
    # f_json.close()
