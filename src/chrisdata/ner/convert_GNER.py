import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby, islice
from typing import Optional, Iterable, Tuple, List
from urllib.parse import urljoin

import httpx
import typer
from bs4 import BeautifulSoup

from chrisbase.data import ProjectEnv, InputOption, FileOption, OutputOption, IOArguments, JobTimer, FileStreamer, TableOption, MongoStreamer
from chrisbase.io import LoggingFormat, new_path, merge_dicts
from chrisbase.util import mute_tqdm_cls, shuffled
from . import *

logger = logging.getLogger(__name__)
http_clients: Optional[List[httpx.Client]] = None
file_pattern = re.compile("\[\[File:([^]]+)]]")
space_pattern = re.compile("  +")
link2_pattern = re.compile("\[\[([^|\]]+)\|([^]]+)]]")
link1_pattern = re.compile("\[\[([^]]+)]]")
bold3_pattern = re.compile("'''([^']+)'''")
bold2_pattern = re.compile("''([^']+)''")
special_pattern1 = re.compile("{{.+?}}")
special_pattern2 = re.compile("{{[^}]+?}}")
reference_pattern = re.compile("<ref[^>]*>.*?</ref>")
bio_tag_pattern = re.compile("([^ ]+)\(([BIO](-[A-Za-z ]+)?)\)")


class ExtraOption(BaseModel):
    get_entity_texts_fn: str = "get_entity_texts_from_train"
    min_entity_freq: int = 1
    min_entity_chars: int = 3
    min_entity_links: int = 3
    max_entity_targets: int = 10
    max_search_candidate: int = 5
    max_targets_per_page: int = 3


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


def entity_texts_to_freq_dict(entity_texts: List[str], args: IOArguments):
    # count entity frequency using groupby
    entity_freq = {k: len(list(g)) for k, g in groupby(sorted(entity_texts)) if entity_text_pattern.fullmatch(k)}
    # sort by frequency
    entity_freq = dict(sorted(entity_freq.items(), key=lambda x: x[1], reverse=True))
    # filter out entities with frequency less than min_entity_chars
    entity_freq = {k: v for k, v in entity_freq.items() if v >= args.option.min_entity_freq and len(k) >= args.option.min_entity_chars}
    return entity_freq


def get_entity_texts_from_train(input_file: FileStreamer):
    all_entity_texts = []
    for sample in json.load(input_file.fp):
        sample = GNER_TrainSampleComp.model_validate(sample)
        labels = [g[1] for g in bio_tag_pattern.findall(sample.instance.prompt_labels)]
        words = sample.instance.instruction_inputs.splitlines()[-1].split("Sentence: ")[-1].split()
        if len(words) != len(labels):
            continue
        entities = bio_to_entities(words, labels)
        entity_texts = [entity['text'] for entity in entities]
        all_entity_texts.extend(entity_texts)
    return all_entity_texts


def get_entity_texts_from_test(input_file: FileStreamer):
    all_entity_texts = []
    for sample in input_file:
        sample = json.loads(sample)
        labels = sample['instance']['labels']
        words = sample['instance']['words']
        if len(words) != len(labels):
            continue
        entities = bio_to_entities(words, labels)
        entity_texts = [entity['text'] for entity in entities]
        all_entity_texts.extend(entity_texts)
    return all_entity_texts


def process_one(ii_entity: Tuple[int, str], args: IOArguments) -> Optional[EntityRelatedPassages]:
    ii, entity = ii_entity
    id = f"J{ii:08d}"
    if args.env.calling_sec > 0:
        time.sleep(args.env.calling_sec)
    global http_clients
    http_client = http_clients[ii % len(http_clients)]
    # logger.info(f"- {id} | {http_client._transport._pool._local_address:<15s} | {entity}")

    related_passages = []
    # search on web: https://en.wikipedia.org/w/index.php?fulltext=1&ns0=1&search=[entity_text]
    source_url = f"https://en.wikipedia.org/w/index.php?fulltext=1&ns0=1&search={entity.replace(' ', '+')}"
    try:
        search_results = BeautifulSoup(http_client.get(source_url).text, 'html.parser').select("div.mw-search-result-heading")
    except Exception as e:
        logger.error(f"{type(e).__name__} on http_client.get(source_url): {source_url}")
        return None
    for search_result in search_results[: args.option.max_search_candidate]:
        source_link_url = urljoin(source_url, search_result.select_one("a").attrs['href']).replace('/wiki/', '/w/index.php?title=') + '&action=edit'
        try:
            search_result_page = BeautifulSoup(http_client.get(source_link_url).text, 'html.parser')
        except Exception as e:
            logger.error(f"{type(e).__name__} on http_client.get(source_link_url): {source_link_url}")
            return None
        try:
            document_title = (search_result_page.select_one('#firstHeadingTitle') or search_result_page.select_one('#contentSub')).text
        except Exception as e:
            logger.error(f"{type(e).__name__} on search_result_page.select_one: {source_link_url}")
            return None
        document_content = search_result_page.select_one('#wpTextbox1').text
        document_content = reference_pattern.sub("", document_content)
        document_content = special_pattern1.sub("", document_content)
        document_content = special_pattern2.sub("", document_content)
        document_content = file_pattern.sub("", document_content)
        document_content = space_pattern.sub(" ", document_content)

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
            if f"[[{entity.lower()}]]" not in target.lower():
                continue
            target_lines.append(target)
            # logger.debug(f"- [target] {target}")
        if len(target_lines) == 0:
            continue
        target_lines = shuffled(target_lines)
        target_lines = target_lines[: args.option.max_targets_per_page]
        related_passages.extend(target_lines)

    # for trainable_passage in related_passages:
    #     logger.debug(f"- [trainable_passage] {trainable_passage}")
    return EntityRelatedPassages(
        id=id,
        entity=entity,
        passages=related_passages,
        num_passages=len(related_passages),
        source_url=source_url,
    )


def process_many1(item: Iterable[Tuple[int, str]], args: IOArguments, writer: MongoStreamer, item_is_batch: bool = True):
    inputs = item if item_is_batch else [item]
    if not writer.opt.reset:
        inputs = [x for x in inputs if writer.count({"_id": f"J{x[0]:08d}"}) == 0]
    outputs = [process_one(x, args) for x in inputs]
    outputs = {v.id: v for v in outputs if v}
    outputs = [merge_dicts({"_id": k}, v.model_dump(exclude={"id"})) for k, v in outputs.items() if v]
    if len(outputs) > 0:
        writer.table.insert_many(outputs)


def process_many2(item: Iterable[Tuple[int, str]], args: IOArguments, writer: MongoStreamer, item_is_batch: bool = True):
    inputs = item if item_is_batch else [item]
    if not writer.opt.reset:
        inputs = [x for x in inputs if writer.count({"_id": f"J{x[0]:08d}"}) == 0]
    with ProcessPoolExecutor(max_workers=args.env.max_workers) as exe:
        jobs = [exe.submit(process_one, x, args) for x in inputs]
        outputs = [job.result(timeout=args.env.waiting_sec) for job in jobs]
    outputs = {v.id: v for v in outputs if v}
    outputs = [merge_dicts({"_id": k}, v.model_dump(exclude={"id"})) for k, v in outputs.items() if v]
    if len(outputs) > 0:
        writer.table.insert_many(outputs)


@app.command()
def convert_GNER(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="convert_GNER"),
        logging_home: str = typer.Option(default="output/GNER/convert"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=12),
        debugging: bool = typer.Option(default=False),
        # input
        input_inter: int = typer.Option(default=1),
        input_batch: int = typer.Option(default=10),
        input_file_path: str = typer.Option(default="input/GNER/pile-ner.json"),
        # input_file_path: str = typer.Option(default="input/GNER/zero-shot-test.jsonl"),
        # output
        output_table_reset: bool = typer.Option(default=True),
        output_table_home: str = typer.Option(default="localhost:8800/GNER"),
        output_table_name: str = typer.Option(default="inst_tuning_base-from-train"),
        # output_table_name: str = typer.Option(default="inst_tuning_base-from-test"),
        output_file_path: str = typer.Option(default="output/GNER/convert/train-data.jsonl"),
        # output_file_path: str = typer.Option(default="output/GNER/convert/test-data.jsonl"),
        # option
        get_entity_texts_fn: str = typer.Option(default="get_entity_texts_from_train"),
        # get_entity_texts_fn: str = typer.Option(default="get_entity_texts_from_test"),
        min_entity_freq: int = typer.Option(default=2),
        min_entity_chars: int = typer.Option(default=3),
        min_entity_links: int = typer.Option(default=3),
        max_entity_targets: int = typer.Option(default=70000),
        max_search_candidate: int = typer.Option(default=5),
        max_targets_per_page: int = typer.Option(default=3),
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
        batch=input_batch if not debugging else 1,
        inter=input_inter if not debugging else 1,
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
        table=TableOption(
            home=output_table_home,
            name=output_table_name,
            reset=output_table_reset,
            required=True,
        ),
    )
    extra_opt = ExtraOption(
        get_entity_texts_fn=get_entity_texts_fn,
        min_entity_freq=min_entity_freq,
        min_entity_chars=min_entity_chars,
        min_entity_links=min_entity_links,
        max_entity_targets=max_entity_targets,
        max_search_candidate=max_search_candidate,
        max_targets_per_page=max_targets_per_page,
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
    logging.getLogger("httpx").setLevel(logging.WARNING)

    global http_clients
    http_clients = [
        httpx.Client(
            transport=httpx.HTTPTransport(local_address=ip_addr),
            timeout=httpx.Timeout(timeout=120.0),
        ) for ip_addr in env.ip_addrs
    ]

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(args.input.file) as input_file,
        FileStreamer(args.output.file) as output_file,
        MongoStreamer(args.output.table) as output_table,
    ):
        # set entity list
        if args.option.get_entity_texts_fn == "get_entity_texts_from_train":
            entity_texts = get_entity_texts_from_train(input_file)
        elif args.option.get_entity_texts_fn == "get_entity_texts_from_test":
            entity_texts = get_entity_texts_from_test(input_file)
        else:
            raise ValueError(f"Unknown get_entity_texts_fn: {args.option.get_entity_texts_fn}")
        logger.info("Number of entities in entity_texts: %d", len(entity_texts))
        entity_freq = entity_texts_to_freq_dict(entity_texts, args)
        logger.info("Number of entities in entity_freq: %d", len(entity_freq))
        entity_list = sorted(islice(shuffled(entity_freq.keys()), args.option.max_entity_targets), key=lambda x: str(x).upper())
        logger.info("Number of entities in entity_list: %d", len(entity_list))
        input_data = args.input.ready_inputs(enumerate(entity_list, start=1), total=len(entity_list))

        # process loop
        with tqdm(total=input_data.num_item, unit="item", pre="=>", desc="converting", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
            for item in input_data.items:
                # logger.info(f"args.env.max_workers={args.env.max_workers}")
                if args.env.max_workers <= 1:
                    # logger.info("work with process_many1")
                    process_many1(item=item, args=args, writer=output_table, item_is_batch=input_data.has_batch_items())
                else:
                    # logger.info("work with process_many2")
                    process_many2(item=item, args=args, writer=output_table, item_is_batch=input_data.has_batch_items())
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)

        # export loop
        with tqdm(total=len(output_table), unit="row", pre="=>", desc="exporting", unit_divisor=args.input.inter * 10) as prog:
            for row in output_table:
                output_file.fp.write(json.dumps(row, ensure_ascii=False) + '\n')
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
            logger.info(f"Export {prog.n}/{len(entity_list)} rows to [{output_file.opt}]")

    for http_client in http_clients:
        http_client.close()
