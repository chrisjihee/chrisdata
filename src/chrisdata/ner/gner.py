import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from enum import Enum
from itertools import groupby, islice
from typing import Iterable
from urllib.parse import urljoin

import httpx
import typer
import yaml
from bs4 import BeautifulSoup
from chrisbase.data import ProjectEnv, InputOption, FileOption, OutputOption, IOArguments, JobTimer, FileStreamer, TableOption, MongoStreamer, NewProjectEnv
from chrisbase.io import LoggingFormat, new_path, merge_dicts, normalize_simple_list_in_json, LoggerWriter, dirs, text_blocks, all_line_list
from chrisbase.util import mute_tqdm_cls, shuffled
from progiter import ProgIter
from typing_extensions import Annotated

from . import *

logger = logging.getLogger(__name__)


class ExtraOption(BaseModel):
    min_entity_freq: int = 1
    min_entity_chars: int = 3
    min_entity_links: int = 3
    max_entity_targets: int = 10
    max_search_candidate: int = 5
    max_targets_per_page: int = 3
    max_passage_length: int = 1000
    min_passage_length: int = 100


class EntityTextSource(str, Enum):
    JSON = ".JSON"
    JSONL = ".JSONL"


def ner_samples_from_json(input_file: FileStreamer) -> Iterable[GenNERSampleWrapper]:
    for sample in json.load(input_file.fp):
        sample = GenNERSampleWrapper.model_validate(sample)
        sample.id = sample.instance.id = sample.id or sample.instance.id
        if not sample.instance.words or not sample.instance.labels or not sample.label_list:
            sample.set_missing_values_by_instruction_prompt(input_file.path)
        yield sample


def ner_samples_from_jsonl(input_file: FileStreamer) -> Iterable[GenNERSampleWrapper]:
    for sample in input_file:
        sample = GenNERSampleWrapper.model_validate_json(sample)
        sample.id = sample.instance.id = sample.id or sample.instance.id
        if not sample.instance.words or not sample.instance.labels or not sample.label_list:
            sample.set_missing_values_by_instruction_prompt(input_file.path)
        yield sample


def ner_samples(input_file: FileStreamer) -> Iterable[GenNERSampleWrapper]:
    suffix = input_file.path.suffix.upper()
    if suffix == EntityTextSource.JSON:
        return ner_samples_from_json(input_file)
    elif suffix == EntityTextSource.JSONL:
        return ner_samples_from_jsonl(input_file)
    else:
        raise ValueError(f"Unsupported suffix: {suffix}")


def valid_words_labels(input_file: FileStreamer):
    for sample in ner_samples(input_file):
        if len(sample.instance.words) == len(sample.instance.labels):
            yield sample.instance.words, sample.instance.labels


def get_entity_texts(input_file: FileStreamer):
    all_entity_texts = []
    for words, labels in valid_words_labels(input_file):
        entities = bio_to_entities(words, labels)
        entity_texts = [entity['text'] for entity in entities]
        all_entity_texts.extend(entity_texts)
    return all_entity_texts


def entity_texts_to_freq_dict(entity_texts: List[str], extra_opt: ExtraOption):
    # count entity frequency using groupby
    entity_freq = {k: len(list(g)) for k, g in groupby(sorted(entity_texts)) if entity_text_pattern.fullmatch(k)}
    # sort by frequency
    entity_freq = dict(sorted(entity_freq.items(), key=lambda x: x[1], reverse=True))
    # filter out entities with frequency less than min_entity_chars
    entity_freq = {k: v for k, v in entity_freq.items() if v >= extra_opt.min_entity_freq and len(k) >= extra_opt.min_entity_chars}
    return entity_freq


def process_one(idx_entity: Tuple[int, str], args: IOArguments) -> Optional[EntityRelatedPassages]:
    idx, entity = idx_entity
    job_id = f"J{idx:08d}"
    if args.env.calling_sec > 0:
        time.sleep(args.env.calling_sec)
    extra_opt: ExtraOption = ExtraOption.model_validate(args.option)
    http_client: httpx.Client = args.env.http_clients[idx]
    # logger.info(f"- {id} | {http_client._transport._pool._local_address:<15s} | {entity}")

    all_entity_passages = []
    # search on web: https://en.wikipedia.org/w/index.php?fulltext=1&ns0=1&search=[entity_text]
    source_url = f"https://en.wikipedia.org/w/index.php?fulltext=1&ns0=1&search={entity.replace(' ', '+')}"
    try:
        search_results = BeautifulSoup(http_client.get(source_url).text, 'html.parser').select("div.mw-search-result-heading")
    except Exception as e:
        logger.error(f"{type(e).__name__} on http_client.get(source_url): {source_url}")
        return None
    for search_result in search_results[: extra_opt.max_search_candidate]:
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

        document_passages = []
        title_pattern = re.compile(rf"\b{re.escape(document_title)}\b")
        for origin in document_content.splitlines():
            target = origin
            target = title_pattern.sub(f"[[{document_title}]]", target)
            target = link2_pattern.sub(r"[[\2]]", target)
            target = bold3_pattern.sub(r"\1", target)
            target = bold2_pattern.sub(r"\1", target)
            if extra_opt.min_passage_length >= 0:
                if len(target) < extra_opt.min_passage_length:
                    continue
            if extra_opt.max_passage_length >= 0:
                if len(target) > extra_opt.max_passage_length:
                    continue
            if f"[[{entity.lower()}]]" not in target.lower():
                continue
            if len(link1_pattern.findall(target)) < extra_opt.min_entity_links:
                continue
            document_passages.append(target)
            # logger.info(f"- [passage]({len(target)}) {target}")
        if len(document_passages) > 0:
            document_passages = shuffled(document_passages)[: extra_opt.max_targets_per_page]
            all_entity_passages.extend(document_passages)

    return EntityRelatedPassages(
        id=job_id,
        entity=entity,
        passages=all_entity_passages,
        num_passages=len(all_entity_passages),
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


@app.command("crawl_wiki")
def crawl_wiki_by_entity(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="crawl_wiki_by_entity"),
        logging_home: str = typer.Option(default="output/GNER"),
        logging_file: str = typer.Option(default="crawl_wiki.out"),
        max_workers: int = typer.Option(default=12),
        debugging: bool = typer.Option(default=False),
        # input
        input_inter: int = typer.Option(default=1),
        input_batch: int = typer.Option(default=10),
        input_file: str = typer.Argument(default=...),
        # input_file = "GNER/data/pile-ner.json"
        # input_file = "GNER/data/zero-shot-test.jsonl"
        # output
        output_file: str = typer.Argument(default=...),
        # output_file = "output/GNER/wiki_passage_from_pile.jsonl"
        # output_file = "output/GNER/wiki_passage_from_zero.jsonl"
        output_table_home: str = typer.Option(default="localhost:8800/GNER"),
        output_table_reset: bool = typer.Option(default=True),
        # option
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
            path=input_file,
            required=True,
        ),
    )
    output_opt = OutputOption(
        file=FileOption.from_path(
            path=output_file,
            name=new_path(output_file, post=env.time_stamp).name,
            mode="w",
        ),
        table=TableOption(
            home=output_table_home,
            name=Path(output_file).stem,
            reset=output_table_reset,
            required=True,
        ),
    )
    extra_opt = ExtraOption(
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
    assert args.output.table, "output.table is required"
    logging.getLogger("httpx").setLevel(logging.WARNING)

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(args.input.file) as input_file,
        FileStreamer(args.output.file) as output_file,
        MongoStreamer(args.output.table) as output_table,
    ):
        # set entity list
        entity_texts = get_entity_texts(input_file)
        logger.info("Number of entities in entity_texts: %d", len(entity_texts))
        entity_freq = entity_texts_to_freq_dict(entity_texts, extra_opt)
        logger.info("Number of entities in entity_freq: %d", len(entity_freq))
        entity_list = sorted(islice(shuffled(entity_freq.keys()), extra_opt.max_entity_targets), key=lambda x: str(x).upper())
        logger.info("Number of entities in entity_list: %d", len(entity_list))
        input_data = args.input.ready_inputs(enumerate(entity_list, start=1), total=len(entity_list))

        # process loop
        with tqdm(total=input_data.num_item, unit="item", pre="=>", desc="crawling", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
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


@app.command("convert_wiki")
def convert_wiki_to_jsonl(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="convert_wiki_to_jsonl"),
        logging_home: str = typer.Option(default="output/GNER"),
        logging_file: str = typer.Option(default="convert_wiki.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),
        # input
        input_inter: int = typer.Option(default=1000),
        input_file: str = typer.Argument(default=...),
        # input_file = "output/GNER/wiki_passage_from_pile.jsonl"
        # input_file = "output/GNER/wiki_passage_from_zero.jsonl"
        instruction_file: str = typer.Option(default="GNER/configs/instruction_configs/instruction.json"),
        # output
        output_file: str = typer.Argument(default=...),
        # output_file = "GNER/data/linked-entity-pile.jsonl"
        # output_file = "GNER/data/linked-entity-zero.jsonl"
        # option
        split_name: str = typer.Option(default="train"),
        label_name: str = typer.Option(default="LinkedEntity"),
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
        inter=input_inter if not debugging else 1,
        file=FileOption.from_path(
            path=input_file,
            required=True,
        ),
    )
    output_opt = OutputOption(
        file=FileOption.from_path(
            path=output_file,
            name=new_path(output_file, post=env.time_stamp).name,
            mode="w",
        ),
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.file, "input.file is required"
    assert args.output.file, "output.file is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(args.input.file) as input_file,
        FileStreamer(args.output.file) as output_file,
    ):
        logger.info("split_name: %s", split_name)
        logger.info("label_name: %s", label_name)

        all_passages = []
        for row in input_file:
            all_passages.extend(EntityRelatedPassages.model_validate_json(row).passages)
        logger.info("Number of passages in all_passages: %d", len(all_passages))

        label_list = [label_name]
        with tqdm(total=len(all_passages), unit="item", pre="=>", desc="converting", unit_divisor=args.input.inter) as prog:
            for i, wiki_passage in enumerate(all_passages):
                instance_id = f"new_{i}"
                instance = GenNERSample.from_wiki_passage(
                    wiki_passage=wiki_passage,
                    label=label_name,
                    id=instance_id,
                ).set_instruction_prompt(
                    instruction_file=instruction_file,
                    label_list=label_list,
                )
                wrapped = GenNERSampleWrapper(
                    id=instance_id,
                    dataset=input_file.path.stem,
                    split=split_name,
                    label_list=label_list,
                    instance=instance,
                )
                output_file.fp.write(wrapped.model_dump_json() + "\n")
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
        logger.info("Number of samples in output_file: %d", prog.n)


@app.command("convert_message")
def convert_message_to_jsonl(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="convert_message_to_jsonl"),
        logging_home: str = typer.Option(default="output/GNER"),
        logging_file: str = typer.Option(default="convert_message.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),
        # input
        input_inter: int = typer.Option(default=1000),
        input_total: int = typer.Option(default=53220),
        # input_file: str = typer.Argument(default=...),
        input_file: str = typer.Argument(default="LLM-based/generation/YAGO3-10/edges_as_text_all-messages-53220@3.jsonl"),
        instruction_file: str = typer.Argument(default="LLM-based/template/generation-KG.txt"),
        # output
        # output_file: str = typer.Argument(default=...),
        output_file: str = typer.Argument(default="GNER/data/KG-generation-YAGO3-53220@3.jsonl"),
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
        inter=input_inter if not debugging else 1,
        file=FileOption.from_path(
            path=input_file,
            required=True,
        ),
    )
    output_opt = OutputOption(
        file=FileOption.from_path(
            path=output_file,
            name=new_path(output_file, post=env.time_stamp).name,
            mode="w",
        ),
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.file, "input.file is required"
    assert args.output.file, "output.file is required"
    instruction_template = Path(instruction_file).read_text()
    response_template = "<generation>" + instruction_template.split("<generation>")[-1]

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(args.input.file) as input_file,
        FileStreamer(args.output.file) as output_file,
    ):
        with tqdm(total=input_total, unit="item", pre="=>", desc="converting", unit_divisor=args.input.inter) as prog:
            for sample_idx, sample in enumerate(input_file):
                sample = KGGenerationMessage.model_validate_json(sample)
                # instruction = [x.content for x in sample.generation_messages][-1]
                instruction = "\n".join([x.content for x in sample.generation_messages])
                prompt_labels = response_template.format(
                    generation_form=normalize_simple_list_in_json(json.dumps(
                        {
                            "target_entity": sample.entity,
                            "triples_by_model": sample.triples_by_human,
                            "number_of_triples": len(sample.triples_by_human),
                        }, indent=2, ensure_ascii=False,
                    ))
                )
                instance_id = f"new_{sample_idx}"
                wrapped = GenSeq2SeqSampleWrapper(
                    id=instance_id,
                    dataset=sample.dataset_name,
                    split="train",
                    instance=GenSeq2SeqSample(
                        id=instance_id,
                        instruction_inputs=instruction,
                        prompt_labels=prompt_labels
                    )
                )
                output_file.fp.write(wrapped.model_dump_json() + "\n")
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
        logger.info("Number of samples in output_file: %d", prog.n)


@app.command("convert_json")
def convert_json_to_jsonl(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="convert_json_to_jsonl"),
        logging_home: str = typer.Option(default="output/GNER"),
        logging_file: str = typer.Option(default="convert_json.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),
        # input
        input_inter: int = typer.Option(default=10000),
        input_total: int = typer.Option(default=105659),
        input_file: str = typer.Argument(default=...),
        # input_file = "input/GNER/pile-ner.json"
        # output
        output_file: str = typer.Argument(default=...),
        # output_file = "input/GNER/pile-ner.jsonl"
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
        inter=input_inter if not debugging else 1,
        file=FileOption.from_path(
            path=input_file,
            required=True,
        ),
    )
    output_opt = OutputOption(
        file=FileOption.from_path(
            path=output_file,
            name=new_path(output_file, post=env.time_stamp).name,
            mode="w",
        ),
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.file, "input.file is required"
    assert args.output.file, "output.file is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(args.input.file) as input_file,
        FileStreamer(args.output.file) as output_file,
    ):
        with tqdm(total=input_total, unit="item", pre="=>", desc="converting", unit_divisor=args.input.inter) as prog:
            for sample in ner_samples(input_file):
                output_file.fp.write(sample.model_dump_json() + "\n")
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
        logger.info("Number of samples in output_file: %d", prog.n)


conll_label = re.compile("[ \t](O|[BIES]-[^\n]+)$")


def read_class_names(input_file):
    all_label_names = []
    for text_block in text_blocks(input_file):
        for line in text_block:
            label = conll_label.search(line).group(1)
            if label not in all_label_names:
                all_label_names.append(label)
    all_class_names = []
    for label_name in all_label_names:
        class_name = re.sub(r"^[BIES]-|^O$", "", label_name)
        if class_name and class_name not in all_class_names:
            all_class_names.append(class_name)
    return sorted(all_class_names)


def normalize_conll(input_file, temp_file="temp.txt"):
    with Path(temp_file).open("w") as f:
        for text_block in text_blocks(input_file):
            for line in text_block:
                m = conll_label.search(line)
                assert m, f"Invalid line: {line}"
                word = line[:m.start()]
                if len(word) == 0:
                    word = " "
                assert word, f"Invalid word: input_file={input_file}, text_block=[{text_block}] / len(word)={len(word)}"
                label = m.group(1)
                label = label.replace(" ", "_").upper()  # for easy post-processing
                label = re.sub("^S-", "B-", label)  # normalize to BIO-style
                label = re.sub("^E-", "I-", label)  # normalize to BIO-style
                f.write(f"{word}\t{label}\n")
            f.write("\n")
    with Path(temp_file).open() as f:
        with Path(input_file).open("w") as g:
            for line in f:
                g.write(line)


def verify_conll(input_file, label_names):
    for text_block in text_blocks(input_file):
        for line in text_block:
            m = conll_label.search(line)
            assert m, f"Invalid line: {line}"
            word = line[:m.start()]
            if len(word) == 0:
                word = " "
            label = m.group(1)
            assert word, f"Invalid word: input_file={input_file}, text_block=[{text_block}] / len(word)={len(word)}"
            assert label in label_names, f"Invalid label: {label}, label_names={label_names}"


@app.command("normalize_conll")
def normalize_conll_dirs(
        input_dirs: Annotated[str, typer.Argument()] = "data/GNER/*",  # "data/GNER/*"
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    env = NewProjectEnv(logging_level=logging_level)
    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
    ):
        for input_dir in sorted(dirs(input_dirs), key=lambda x: x.name.lower()):
            logger.info("[input_dir]: %s", input_dir)
            train_file = input_dir / "train.txt"
            eval_file = input_dir / "dev.txt"
            test_file = input_dir / "test.txt"
            assert train_file.exists() and eval_file.exists() and test_file.exists()
            assert train_file.is_file() and eval_file.is_file() and test_file.is_file()

            for input_file in [train_file, eval_file, test_file]:
                normalize_conll(input_file)

            class_names = read_class_names(train_file)
            label_names = [f"B-{x}" for x in class_names] + [f"I-{x}" for x in class_names] + ["O"]
            logger.info("  - class(%d): %s", len(class_names), ', '.join(class_names))
            logger.info("  - label(%d): %s", len(label_names), ', '.join(label_names))

            for input_file in [train_file, eval_file, test_file]:
                verify_conll(input_file, label_names)

            (input_dir / "label.txt").write_text("\n".join(class_names) + "\n")


@app.command("sample_jsonl")
def sample_jsonl_lines(
        input_file: Annotated[str, typer.Argument()] = ...,
        output_file: Annotated[str, typer.Option("--output_file")] = "",
        num_samples: Annotated[int, typer.Option("--num_samples")] = 100,
        verbose: Annotated[int, typer.Option("--verbose")] = 2,
):
    env = NewProjectEnv()
    with JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=verbose >= 1):
        with Path(input_file).open() as input_fp:
            input_lines = [x.strip() for x in input_fp.readlines() if x.strip()]
            if len(input_lines) > num_samples:
                sampled_indices = sorted(random.sample(range(len(input_lines)), num_samples))
                input_lines = [input_lines[i] for i in sampled_indices]
        num_samples = len(input_lines)
        if not output_file:
            output_file = new_path(input_file, post=num_samples, sep='=')
        with Path(output_file).open("w", encoding="utf-8") as output_fp:
            num_outputs = 0
            for x in input_lines:
                output_fp.write(x + "\n")
                num_outputs += 1
        logger.info(f"Number of samples in {output_file}: %d", num_outputs)


@app.command("stratified_sample_jsonl")
def stratified_sample_jsonl(
        input_file: Annotated[str, typer.Argument()] = ...,  # "data/pile-ner.jsonl",
        output_file: Annotated[str, typer.Option("--output_file")] = None,
        min_num_word: Annotated[int, typer.Option("--min_num_word")] = 0,
        max_num_word: Annotated[int, typer.Option("--max_num_word")] = 512,
        min_num_label: Annotated[int, typer.Option("--min_num_label")] = 0,
        max_num_label: Annotated[int, typer.Option("--max_num_label")] = 300,
        min_num_samples: Annotated[int, typer.Option("--min_num_samples")] = 0,
        max_num_samples: Annotated[int, typer.Option("--max_num_samples")] = 10000,
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
        random_seed: Annotated[int, typer.Option("--random_seed")] = 7,
):
    env = NewProjectEnv(logging_level=logging_level, random_seed=random_seed)
    output_file = Path(output_file) if output_file else new_path(input_file, post="sampled")
    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
        FileStreamer(FileOption.from_path(path=input_file, required=True)) as input_file,
        FileStreamer(FileOption.from_path(path=output_file, mode="w")) as output_file,
    ):
        random.seed(env.random_seed)
        data_label_lists = dict()
        for sample in ProgIter(ner_samples(input_file), total=len(input_file), desc=f"Sampling {input_file.path}:", stream=LoggerWriter(logger, level=logging_level), verbose=3):
            sample.label_list = [str(x).replace(" ", "_").upper() for x in sample.label_list]  # for easy post-processing
            sample.instance.labels = [str(x).replace(" ", "_").upper() for x in sample.instance.labels]  # for easy post-processing
            if len(sample.instance.words) != len(sample.instance.labels):
                continue
            possible_labels = [tag for entity_type in sample.label_list for tag in (f"B-{entity_type}", f"I-{entity_type}")] + ["O"]
            if any(label not in possible_labels for label in sample.instance.labels):
                continue
            if min_num_word <= len(sample.instance.words) == len(sample.instance.labels) <= max_num_word:
                label_list = sorted(set(sample.label_list))
                if min_num_label <= len(label_list) <= max_num_label:
                    data_label_list = f'{len(label_list):04d} {" ".join(label_list)}'
                    data_label_lists.setdefault(data_label_list, []).append(sample)
        for data_label_list in sorted(data_label_lists.keys()):
            if min_num_samples <= len(data_label_lists[data_label_list]):
                if max_num_samples < len(data_label_lists[data_label_list]):
                    sampled_indices = sorted(random.sample(range(len(data_label_lists[data_label_list])), max_num_samples))
                    data_label_lists[data_label_list] = [data_label_lists[data_label_list][i] for i in sampled_indices]
            else:
                del data_label_lists[data_label_list]
        num_outputs = 0
        for samples in data_label_lists.values():
            for sample in samples:
                output_file.fp.write(sample.model_dump_json() + "\n")
                num_outputs += 1
        logger.info(f"Number of samples in {output_file.path}: %d", num_outputs)
        final_output_file = new_path(output_file.path, post=f"N{num_outputs}")
        logger.info(f"Renamed output file to {final_output_file}")
    output_file.path.replace(final_output_file)
    print()
    return final_output_file


@app.command("convert_to_hybrid_round_cot_version")
def convert_to_hybrid_round_cot_version(
        input_file: Annotated[str, typer.Argument()] = ...,  # "data/GoLLIE/baseline/ace05.ner.dev.jsonl",
        output_file: Annotated[str, typer.Option("--output_file")] = None,
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    env = NewProjectEnv(logging_level=logging_level)
    output_file = Path(output_file) if output_file else new_path(input_file, post="GNER")
    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
        FileStreamer(FileOption.from_path(path=input_file, required=True)) as input_file,
        FileStreamer(FileOption.from_path(path=output_file, mode="w")) as output_file,
    ):
        for line in islice(input_file, 0, 10):
            print(f"[line] {line}")
            sample = GoLLIESample.model_validate_json(line)
            print(f"[sample.ids] {sample.ids}")
            print(f"[sample.task_id] {sample.task_id}")
            print(f"[sample.scorer_cls] {sample.scorer_cls}")
            print(f"[sample.labels] {sample.labels}")
            print(f"[sample.unlabelled_sentence] {sample.unlabelled_sentence}")
            print(f"[sample.text] {sample.text}")
            print(f"[yaml.safe_load(sample.txt)] {yaml.safe_load(sample.text)}")
            print("=" * 40)
            exit(1)
        exit(1)


@app.command("convert_to_hybrid_round_version")
def convert_to_hybrid_round_version(
        mr_input_file: Annotated[Optional[str], typer.Option("--mr_input_file")] = None,  # "data/pile-ner=10-100,3-7,3-10.jsonl", "data/pile-ner=10-100,3-10,3-10.jsonl", "data/pile-ner.jsonl", "data/ZSE-validation.jsonl", "data/ZSE-test.jsonl"
        sr_input_file: Annotated[Optional[str], typer.Option("--sr_input_file")] = None,  # "data/pile-ner.jsonl"
        mr_inst_file: Annotated[Optional[str], typer.Option("--mr_inst_file")] = "configs/instruction/GNER-EQ-MR.txt",  # "configs/instruction/GNER-EQ-MR.txt",
        sr_inst_file: Annotated[Optional[str], typer.Option("--sr_inst_file")] = "configs/instruction/GNER-EQ-SR.txt",  # "configs/instruction/GNER-EQ-SR.txt"
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    assert sr_input_file or mr_input_file, "Either sr_input_file or mr_input_file is required"
    post = "HR" if sr_inst_file and mr_inst_file else "MR" if mr_inst_file else "SR" if sr_inst_file else None
    env = NewProjectEnv(logging_level=logging_level)
    output_file = Path(mr_input_file or sr_input_file)
    output_file = new_path(output_file.with_stem(output_file.stem.split("-N")[0]), post=post)
    mr_inst_temp = Path(mr_inst_file).read_text() if mr_inst_file else None
    sr_inst_temp = Path(sr_inst_file).read_text() if sr_inst_file else None
    if not sr_input_file:
        sr_input_file = mr_input_file
    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
        FileStreamer(FileOption.from_path(path=mr_input_file, required=True)) if mr_input_file else nullcontext() as mr_input_file,
        FileStreamer(FileOption.from_path(path=sr_input_file, required=True)) if sr_input_file else nullcontext() as sr_input_file,
        FileStreamer(FileOption.from_path(path=output_file, mode="w")) as output_file,
    ):
        num_new_sr_samples = 0
        num_new_mr_samples = 0
        logger.info("[output_file]   : %s", output_file.path)
        if mr_input_file:
            logger.info("[mr_input_file] : %s", mr_input_file.path)
        if sr_input_file:
            logger.info("[sr_input_file] : %s", sr_input_file.path)
        if mr_inst_file:
            logger.info("[mr_inst_file]  : %s", mr_inst_file)
        if sr_inst_file:
            logger.info("[sr_inst_file]  : %s", sr_inst_file)

        if sr_input_file:
            for sample in ProgIter(ner_samples(sr_input_file), total=len(sr_input_file), desc=f"Converting {sr_input_file.path}:", stream=LoggerWriter(logger, level=logging_level), verbose=3):
                sample.instance.id = sample.id = sample.instance.id or sample.id
                sample.label_list = [str(x).replace(" ", "_").upper() for x in sample.label_list]  # for easy post-processing
                sample.instance.labels = [str(x).replace(" ", "_").upper() for x in sample.instance.labels]  # for easy post-processing
                if len(sample.instance.words) != len(sample.instance.labels):
                    continue
                possible_labels = [tag for entity_type in sample.label_list for tag in (f"B-{entity_type}", f"I-{entity_type}")] + ["O"]
                if any(label not in possible_labels for label in sample.instance.labels):
                    continue
                sentence = " ".join(sample.instance.words)
                logger.debug("\n" * 5)
                logger.debug(f">> old_sample_id={sample.id}")
                logger.debug(f">> old_instruction_inputs=\n{sample.instance.instruction_inputs}")
                logger.debug(f">> old_prompt_labels=\n{sample.instance.prompt_labels}")

                if sr_inst_temp:
                    entity_types = ", ".join(sample.label_list)
                    possible_labels = [tag for entity_type in sample.label_list for tag in (f"B-{entity_type}", f"I-{entity_type}")] + ["O"]
                    final_words, final_labels = sample.instance.words, [x if x in possible_labels else "O" for x in sample.instance.labels]
                    prompt_labels = GenNERSample.get_prompt_labels(final_words, final_labels)
                    instruction_inputs = sr_inst_temp.format(entity_types=entity_types, sentence=sentence)
                    logger.debug("\n" * 2)
                    logger.debug("=" * 80)
                    logger.debug(f">> new_instruction_inputs=\n{'-' * 80}\n{instruction_inputs}\n{'-' * 80}")
                    logger.debug(f">> new_prompt_labels=\n{prompt_labels}")
                    new_sample = GenNERSampleWrapper(
                        id=f"{sample.id}.S",
                        dataset=sample.dataset,
                        split=sample.split,
                        label_list=sample.label_list,
                        instance=GenNERSample(
                            id=f"{sample.id}.S",
                            group=f"{sample.id}",
                            words=final_words,
                            labels=final_labels,
                            target_label="*",
                            prompt_labels=prompt_labels,
                            instruction_inputs=instruction_inputs,
                        )
                    )
                    output_file.fp.write(new_sample.model_dump_json() + "\n")
                    num_new_sr_samples += 1

        if mr_input_file:
            for sample in ProgIter(ner_samples(mr_input_file), total=len(mr_input_file), desc=f"Converting {mr_input_file.path}:", stream=LoggerWriter(logger, level=logging_level), verbose=3):
                sample.instance.id = sample.id = sample.instance.id or sample.id
                sample.label_list = [str(x).replace(" ", "_").upper() for x in sample.label_list]  # for easy post-processing
                sample.instance.labels = [str(x).replace(" ", "_").upper() for x in sample.instance.labels]  # for easy post-processing
                if len(sample.instance.words) != len(sample.instance.labels):
                    continue
                possible_labels = [tag for entity_type in sample.label_list for tag in (f"B-{entity_type}", f"I-{entity_type}")] + ["O"]
                if any(label not in possible_labels for label in sample.instance.labels):
                    continue
                sentence = " ".join(sample.instance.words)
                logger.debug("\n" * 5)
                logger.debug(f">> old_sample_id={sample.id}")
                logger.debug(f">> old_instruction_inputs=\n{sample.instance.instruction_inputs}")
                logger.debug(f">> old_prompt_labels=\n{sample.instance.prompt_labels}")

                for i, entity_type in enumerate(sample.label_list if mr_inst_temp else [], start=1):
                    possible_labels = [tag for tag in (f"B-{entity_type}", f"I-{entity_type}")] + ["O"]
                    final_words, final_labels = sample.instance.words, [x if x in possible_labels else "O" for x in sample.instance.labels]
                    prompt_labels = GenNERSample.get_prompt_labels(final_words, final_labels)
                    instruction_inputs = mr_inst_temp.format(entity_type=entity_type, sentence=sentence)
                    logger.debug("\n" * 2)
                    logger.debug("=" * 80)
                    logger.debug(f">> new_instruction_inputs=\n{'-' * 80}\n{instruction_inputs}\n{'-' * 80}")
                    logger.debug(f">> new_prompt_labels=\n{prompt_labels}")
                    new_sample = GenNERSampleWrapper(
                        id=f"{sample.id}.M{i}",
                        dataset=sample.dataset,
                        split=sample.split,
                        label_list=sample.label_list,
                        instance=GenNERSample(
                            id=f"{sample.id}.M{i}",
                            group=f"{sample.id}",
                            words=final_words,
                            labels=final_labels,
                            target_label=entity_type,
                            prompt_labels=prompt_labels,
                            instruction_inputs=instruction_inputs,
                        )
                    )
                    output_file.fp.write(new_sample.model_dump_json() + "\n")
                    num_new_mr_samples += 1

        logger.warning(f">> Number of new SR samples in {output_file.path} = {num_new_sr_samples}")
        logger.warning(f">> Number of new MR samples in {output_file.path} = {num_new_mr_samples}")
        final_output_file = output_file.path.with_stem(output_file.path.stem.replace(post, f"{post}{num_new_sr_samples + num_new_mr_samples}"
                                                                                           f"{f',{num_new_sr_samples}' if post == 'HR' else ''}"
                                                                                           f"{f',{num_new_mr_samples}' if post == 'HR' else ''}"))
        logger.info(f"Renamed output file to {final_output_file}")
    output_file.path.replace(final_output_file)
    print()
    return final_output_file


def make_prompt_label(sample: GenNERSampleWrapper, word_id: int, level_main: int, level_sub: int):
    if level_main == 1:
        prompt_label = sample.instance.labels[word_id]
    elif level_main == 2:
        prompt_label = GenNERSample.get_prompt_labels([sample.instance.words[word_id]], [sample.instance.labels[word_id]])
    elif level_main == 3:
        total_labels = len(sample.instance.labels)
        start_idx = max(0, word_id)
        end_idx = min(total_labels, word_id + 1)
        labels = (
                ["?"] * start_idx +
                sample.instance.labels[start_idx: end_idx] +
                ["?"] * (total_labels - end_idx)
        )
        assert len(labels) == len(sample.instance.labels)
        prompt_label = GenNERSample.get_prompt_labels(sample.instance.words, labels)
    elif level_main == 4:
        total_labels = len(sample.instance.labels)
        start_idx = max(0, word_id - level_sub)
        end_idx = min(total_labels, word_id + level_sub + 1)
        labels = (
                ["?"] * start_idx +
                sample.instance.labels[start_idx: end_idx] +
                ["?"] * (total_labels - end_idx)
        )
        assert len(labels) == len(sample.instance.labels)
        prompt_label = GenNERSample.get_prompt_labels(sample.instance.words, labels)
    elif level_main == 5:
        prompt_label = GenNERSample.get_prompt_labels(sample.instance.words, sample.instance.labels)
    else:
        raise ValueError(f"Unsupported level_main: {level_main}")
    return prompt_label


@app.command("convert_to_WQ")
def convert_to_word_query_version(
        input_file: Annotated[str, typer.Argument()] = ...,  # "data/pile-ner=10-100,3-7,3-10.jsonl"
        label_level_main: Annotated[int, typer.Option("--label_level_main")] = ...,
        label_level_sub: Annotated[int, typer.Option("--label_level_sub")] = 0,
        instruction_file: Annotated[str, typer.Option("--instruction_file")] = "configs/instruction/GNER-WQ.txt",
        logging_level: Annotated[int, typer.Option("--logging_level")] = logging.INFO,
):
    env = NewProjectEnv(logging_level=logging_level)
    input_dir = Path(input_file).parent
    output_post = f"WQ={label_level_main}{f',{label_level_sub}' if label_level_sub > 0 else ''}"
    output_file = new_path(str(input_dir).replace("-BL", ""), post=output_post) / new_path(input_file, post=output_post).name
    instruction_template = Path(instruction_file).read_text()
    with (
        JobTimer(f"python {env.current_file} {' '.join(env.command_args)}", rt=1, rb=1, rc='=', verbose=logging_level <= logging.INFO),
        FileStreamer(FileOption.from_path(path=input_file, required=True)) as input_file,
        FileStreamer(FileOption.from_path(path=output_file, mode="w")) as output_file,
    ):
        logger.info("[input_file]       : %s", input_file.path)
        logger.info("[output_file]      : %s", output_file.path)
        logger.info("[instruction_file] : %s", instruction_file)
        logger.info("[label_level_main] : %s", label_level_main)
        logger.info("[label_level_sub]  : %s", label_level_sub)
        num_new_samples = 0
        for sample in ProgIter(ner_samples(input_file), total=len(input_file), desc=f"Converting {input_file.path}:",
                               stream=LoggerWriter(logger, level=logging_level), verbose=3):
            sample.instance.id = sample.id = sample.instance.id or sample.id
            sample.label_list = [x.replace(" ", "_") for x in sample.label_list]  # for easy post-processing
            sample.instance.labels = [x.replace(" ", "_") for x in sample.instance.labels]  # for easy post-processing
            if len(sample.instance.words) != len(sample.instance.labels):
                continue
            sentence = " ".join(sample.instance.words)
            label_list = ", ".join(sample.label_list) + " and O."
            logger.debug("\n" * 5)
            logger.debug(f">> old_sample_id={sample.id}")
            logger.debug(f">> old_instruction_inputs=\n{sample.instance.instruction_inputs}")
            logger.debug(f">> old_prompt_labels=\n{sample.instance.prompt_labels}")
            for i, word in enumerate(sample.instance.words):
                instruction_inputs = instruction_template.format(label_list=label_list, sentence=sentence, word=word, position=i)
                prompt_labels = make_prompt_label(sample, i, label_level_main, label_level_sub)
                logger.debug("\n" * 2)
                logger.debug("=" * 80)
                logger.debug(f">> new_instruction_inputs=\n{'-' * 80}\n{instruction_inputs}\n{'-' * 80}")
                logger.debug(f">> new_prompt_labels=\n{prompt_labels}")
                new_sample = GenNERSampleWrapper(
                    id=f"{sample.id}.{i}",
                    dataset=sample.dataset,
                    split=sample.split,
                    label_list=sample.label_list,
                    instance=GenNERSample(
                        id=f"{sample.instance.id}.{i}",
                        group=sample.instance.id,
                        words=sample.instance.words,
                        labels=sample.instance.labels,
                        target_index=i,
                        prompt_labels=prompt_labels,
                        instruction_inputs=instruction_inputs,
                    )
                )
                num_new_samples += 1
                output_file.fp.write(new_sample.model_dump_json() + "\n")
        logger.warning(f">> Number of new samples in {output_file.path} = {num_new_samples}")


sample_X = {
    "id": "ner_0",
    "dataset": "pile-ner",
    "split": "train",
    "instance": {
        "id": "ner_0",
        "prompt_labels": "Q:(O) Position(O) character(O) based(O) on(O) enemy(O) coordinates(O) in(O) lua(B-programming language) I(O) have(O) written(O) a(O) function(B-programming concept) here(O) which(O) should(O) turn(O) my(O) character(O) based(O) on(O) enemy(O) coordinates(O) but(O) it's(O) not(O) perfect(O) because(O) it(O) does(O) not(O) always(O) turn(O) where(O) I(O) want(O) it(O) to(O) and(O) perhaps(O) there(O) is(O) a(O) better(O) way(O) of(O) writing(O) it(O) local(O) myPosition(B-variable) =(O) {x(O) =(O) 350,(O) y(O) =(O) 355}(O) local(O) enemyPosition(B-variable) =(O) {x(O) =(O) 352,(O) y(O) =(O) 354}(O) local(O) xValue,(B-variable) yValue,(B-variable) xDir,(B-variable) yDir,(B-variable) dir(B-variable) if(O) myPosition.x(B-variable) >(O) enemyPosition.x(B-variable) then(O) xValue(B-variable) =(O) myPosition.x(B-variable) -(O)",
        "instruction_inputs": "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n2. I- (Inside) is used for words within a named entity but are not the first word.\n3. O (Outside) denotes words that are not part of a named entity.\n\nUse the specific entity tags: programming concept, programming language, database, variable, Date and O.\nSentence: Q: Position character based on enemy coordinates in lua I have written a function here which should turn my character based on enemy coordinates but it's not perfect because it does not always turn where I want it to and perhaps there is a better way of writing it local myPosition = {x = 350, y = 355} local enemyPosition = {x = 352, y = 354} local xValue, yValue, xDir, yDir, dir if myPosition.x > enemyPosition.x then xValue = myPosition.x -",
        "prediction_output": None,
        "prediction_outputs": None,
        "group": None,
        "words": [
            "Q:", "Position", "character", "based", "on", "enemy", "coordinates", "in", "lua", "I", "have", "written", "a", "function", "here", "which", "should", "turn", "my", "character", "based", "on", "enemy", "coordinates", "but", "it's", "not", "perfect", "because", "it", "does", "not", "always", "turn", "where", "I", "want", "it", "to", "and", "perhaps", "there", "is", "a", "better", "way", "of", "writing", "it", "local", "myPosition", "=", "{x", "=", "350,", "y", "=", "355}", "local",
            "enemyPosition", "=", "{x", "=", "352,", "y", "=", "354}", "local", "xValue,", "yValue,", "xDir,", "yDir,", "dir", "if", "myPosition.x", ">", "enemyPosition.x", "then", "xValue", "=", "myPosition.x", "-"
        ],
        "labels": [
            "O", "O", "O", "O", "O", "O", "O", "O", "B-programming language", "O", "O", "O", "O", "B-programming concept", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-variable", "O", "O", "O", "O", "O", "O", "O", "O", "B-variable", "O", "O", "O", "O", "O", "O", "O", "O", "B-variable", "B-variable", "B-variable", "B-variable", "B-variable", "O", "B-variable",
            "O", "B-variable", "O", "B-variable", "O", "B-variable", "O"
        ],
        "target_index": None,
        "target_label": None
    },
    "label_list": [
        "date", "database", "programming_concept", "programming_language", "variable"
    ]
}

sample_Y = {
    "id": "0.0",
    "dataset": "crossner_ai",
    "split": "train",
    "instance": {
        "id": "0.0",
        "prompt_labels": "[]",
        "instruction_inputs": "Given a sentence, your task is to identify entities for a specific type. Each query asks about one entity type, and the output is a JSON list of entities with their spans (indices in the text).\n* label list:\n['conference', 'metric', 'algorithm', 'country', 'university', 'location', 'programming language', 'organization', 'product', 'researcher', 'task', 'field', 'person']\n* sentence:\nPopular approaches of opinion-based recommender system utilize various techniques including text mining , information retrieval , sentiment analysis ( see also Multimodal sentiment analysis ) and deep learning X.Y. Feng , H. Zhang , Y.J. Ren , P.H. Shang , Y. Zhu , Y.C. Liang , R.C. Guan , D. Xu , ( 2019 ) , , 21 ( 5 ) : e12957 .\n* query:\nIdentify 'conference' entities in the sentence in a JSON list format.\n* output:\n",
        "group": "0",
        "words": ["Popular", "approaches", "of", "opinion-based", "recommender", "system", "utilize", "various", "techniques", "including", "text", "mining", ",", "information", "retrieval", ",", "sentiment", "analysis", "(", "see", "also", "Multimodal", "sentiment", "analysis", ")", "and", "deep", "learning", "X.Y.", "Feng", ",", "H.", "Zhang", ",", "Y.J.", "Ren", ",", "P.H.", "Shang", ",", "Y.", "Zhu", ",", "Y.C.",
                  "Liang", ",", "R.C.", "Guan", ",", "D.", "Xu", ",", "(", "2019", ")", ",", ",", "21", "(", "5", ")", ":", "e12957", "."],
        "labels": ["O", "O", "O", "B-product", "I-product", "I-product", "O", "O", "O", "O", "B-field", "I-field", "O", "B-task", "I-task", "O", "B-task", "I-task", "O", "O", "O", "B-task", "I-task", "I-task", "O", "O", "B-field", "I-field", "B-researcher", "I-researcher", "O", "B-researcher", "I-researcher", "O", "B-researcher", "I-researcher", "O", "B-researcher", "I-researcher", "O", "B-researcher", "I-researcher", "O",
                   "B-researcher", "I-researcher", "O", "B-researcher", "I-researcher", "O", "B-researcher", "I-researcher", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
        "target_word": None,
        "target_label": "conference",
    },
    "label_list": ["conference", "metric", "algorithm", "country", "university", "location", "programming language", "organization", "product", "researcher", "task", "field", "person"]
}
