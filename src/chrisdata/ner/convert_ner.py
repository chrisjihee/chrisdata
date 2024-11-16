import json
import re
from itertools import groupby, islice
from pathlib import Path
from urllib.parse import urljoin

import httpx
import typer
from bs4 import BeautifulSoup

from chrisbase.data import ProjectEnv, InputOption, FileOption, OutputOption, IOArguments, JobTimer, FileStreamer
from chrisbase.io import LoggingFormat, new_path
from chrisbase.util import mute_tqdm_cls, shuffled
from . import *

logger = logging.getLogger(__name__)


class ExtraOption(BaseModel):
    min_entity_freq: int = 1
    min_entity_chars: int = 3
    min_entity_links: int = 3
    max_entity_targets: int = 10
    max_search_candidate: int = 10


def bio_to_entities(words, labels):
    # BIO notation
    pairs = zip(words, labels)
    # convert BIO notation to NER format
    entities = []
    entity = None
    for word, label in pairs:
        if label == 'O':
            entity = None
            continue
        if label.startswith('B-'):
            entity = {'type': label[2:], 'words': [word]}
            entities.append(entity)
        elif label.startswith('I-') and entity:
            entity['words'].append(word)
    for entity in entities:
        entity['text'] = ' '.join(entity.pop('words'))
    return entities


@app.command()
def convert(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="convert_wikidata"),
        logging_home: str = typer.Option(default="output/wikidata/convert"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),
        # input
        input_file_path: str = typer.Option(default="input/GNER/zero-shot-test.jsonl"),
        # output
        output_file_path: str = typer.Option(default="output/GNER/zero-shot-test-conv.jsonl"),
        # option
        min_entity_freq: int = typer.Option(default=2),
        min_entity_chars: int = typer.Option(default=3),
        min_entity_links: int = typer.Option(default=3),
        max_entity_targets: int = typer.Option(default=10),
        max_search_candidate: int = typer.Option(default=5),
):
    env = ProjectEnv(
        project=project,
        job_name=job_name,
        debugging=debugging,
        logging_home=logging_home,
        logging_file=logging_file,
        message_level=logging.INFO,
        message_format=LoggingFormat.CHECK_00,  # if not debugging else LoggingFormat.DEBUG_36,
        max_workers=1 if debugging else max(max_workers, 1),
    )
    input_opt = InputOption(
        file=FileOption.from_path(
            path=input_file_path,
            required=True,
        ),
    )
    output_opt = OutputOption(
        file=FileOption.from_path(
            path=output_file_path,
            name=new_path(output_file_path, post=env.time_stamp).name,
            mode="w",
            required=True,
        ),
    )
    extra_opt = ExtraOption(
        min_entity_freq=min_entity_freq,
        min_entity_chars=min_entity_chars,
        min_entity_links=min_entity_links,
        max_entity_targets=max_entity_targets,
        max_search_candidate=max_search_candidate,
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
        option=extra_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.file, "input.file is required"
    assert args.output.file, "output.file is required"
    reference_pattern = re.compile("<ref[^>]*>.*?</ref>")
    file_pattern = re.compile("\[\[File:([^]]+)]]")
    double_space_pattern = re.compile("  +")
    link2_pattern = re.compile("\[\[([^|\]]+)\|([^]]+)]]")
    link1_pattern = re.compile("\[\[([^]]+)]]")
    bold3_pattern = re.compile("'''([^']+)'''")
    bold2_pattern = re.compile("''([^']+)''")
    special_pattern1 = re.compile("{{.+?}}")
    special_pattern2 = re.compile("{{[^}]+?}}")

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(args.input.file) as input_file,
        FileStreamer(args.output.file) as output_file,
        httpx.Client(timeout=httpx.Timeout(timeout=120.0)) as http_client,
    ):
        input_data = args.input.ready_inputs(input_file, total=len(input_file))
        all_entity_texts = []
        for ii, sample in enumerate([json.loads(a) for a in input_data.items]):
            entities = bio_to_entities(sample['instance']['words'], sample['instance']['labels'])
            entity_texts = [entity['text'] for entity in entities]
            all_entity_texts.extend(entity_texts)
        # count entity frequency using groupby
        entity_freq = {k: len(list(g)) for k, g in groupby(sorted(all_entity_texts))}
        # sort by frequency
        entity_freq = dict(sorted(entity_freq.items(), key=lambda x: x[1], reverse=True))
        # filter out entities with frequency less than 2
        entity_freq = {k: v for k, v in entity_freq.items() if v >= args.option.min_entity_freq and len(k) >= args.option.min_entity_chars}
        # filter out entities containing digits
        entity_freq = {k: v for k, v in entity_freq.items() if not any(char.isdigit() for char in k)}

        for ii, entity_text in enumerate(islice(shuffled(entity_freq.keys()), args.option.max_entity_targets), start=1):
            id = f"{ii:08d}"
            print(f"- {id}: {entity_text}")

            trainable_passages = []
            # search on web: https://en.wikipedia.org/w/index.php?fulltext=1&ns0=1&search=[entity_text]
            entity_search_url = f"https://en.wikipedia.org/w/index.php?fulltext=1&ns0=1&search={entity_text.replace(' ', '+')}"
            response = http_client.get(entity_search_url)
            search_results = BeautifulSoup(response.text, 'html.parser').select("div.mw-search-result-heading")
            for search_result in search_results[: max_search_candidate]:
                search_result_url = urljoin(entity_search_url, search_result.select_one("a").attrs['href']).replace('/wiki/', '/w/index.php?title=') + '&action=edit'
                response = http_client.get(search_result_url)
                search_result_page = BeautifulSoup(response.text, 'html.parser')
                try:
                    document_title = (search_result_page.select_one('#firstHeadingTitle') or search_result_page.select_one('#contentSub')).text
                except:
                    print(f"Error: {search_result_url}")
                    exit(1)
                document_content = search_result_page.select_one('#wpTextbox1').text
                document_content = special_pattern1.sub("", document_content)
                document_content = special_pattern2.sub("", document_content)
                document_content = file_pattern.sub("", document_content)
                document_content = reference_pattern.sub("", document_content)
                document_content = double_space_pattern.sub(" ", document_content)

                # make title pattern with boundary
                title_pattern = re.compile(rf"\b{re.escape(document_title)}\b")
                target_lines = list()
                for origin in document_content.splitlines():
                    target = origin
                    target = title_pattern.sub(f"[[{document_title}]]", target)
                    target = link2_pattern.sub(r"[[\2]]", target)
                    target = bold3_pattern.sub(r"\1", target)
                    target = bold2_pattern.sub(r"\1", target)
                    if len(link1_pattern.findall(target)) < args.option.min_entity_links:
                        continue
                    if f"[[{entity_text.lower()}]]" not in target.lower():
                        continue
                    target_lines.append(target)
                    # print(f"- [target] {target}")
                if len(target_lines) == 0:
                    continue
                trainable_passages.extend(target_lines)

            for trainable_passage in trainable_passages:
                print(f"- [trainable_passage] {trainable_passage}")
