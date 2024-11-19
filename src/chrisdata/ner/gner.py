import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from itertools import groupby, islice
from typing import Optional, Iterable, Tuple
from urllib.parse import urljoin

import httpx
import pandas as pd
import typer
from bs4 import BeautifulSoup

from chrisbase.data import ProjectEnv, InputOption, FileOption, OutputOption, IOArguments, JobTimer, FileStreamer, TableOption, MongoStreamer, CommonArguments
from chrisbase.io import LoggingFormat, new_path, merge_dicts, key_lines, glob_dirs
from chrisbase.util import mute_tqdm_cls, shuffled
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
        if not sample.instance.words or not sample.instance.labels or not sample.label_list:
            sample.set_missing_values_by_instruction_prompt(input_file.path)
        yield sample


def ner_samples_from_jsonl(input_file: FileStreamer) -> Iterable[GenNERSampleWrapper]:
    for sample in input_file:
        sample = GenNERSampleWrapper.model_validate_json(sample)
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


@app.command("convert_conll")
def convert_conll_to_jsonl(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="convert_conll_to_jsonl"),
        logging_home: str = typer.Option(default="output/GNER"),
        logging_file: str = typer.Option(default="convert_conll.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),
        # input
        input_inter: int = typer.Option(default=1000),
        input_dirs: str = typer.Argument(default=...),
        instruction_file: str = typer.Option(default="GNER/configs/instruction_configs/instruction.json"),
        # output
        output_file: str = typer.Argument(default=...),
        # option
        split_name: str = typer.Option(default="train"),  # "train", "dev", "test"
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
            path=input_dirs,
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
    tqdm = mute_tqdm_cls(desc_size=25)
    assert args.input.file, "input.file is required"
    assert args.output.file, "output.file is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(args.input.file) as input_dirs,
        FileStreamer(args.output.file) as output_file,
    ):
        logger.info("split_name: %s", split_name)
        num_outputs = 0
        dataset_builder = GNERDataset()
        for dataset_dir in glob_dirs(input_dirs.path.parent, input_dirs.path.name):
            dataset_path = dataset_dir / Path(split_name).with_suffix(".txt")
            labels_path = dataset_dir / Path("label").with_suffix(".txt")
            if dataset_path.exists() and labels_path.exists():
                instances, label_list = dataset_builder._load_dataset(dataset_path, labels_path)
                with tqdm(total=len(instances), unit="item", pre="=>", desc=dataset_dir.stem, unit_divisor=args.input.inter) as prog:
                    for instance in instances:
                        instance_id = f"{instance.pop('id')}"
                        instance = GenNERSample.model_validate(
                            merge_dicts({"id": f"{instance_id}"}, instance)
                        ).set_instruction_prompt(
                            instruction_file=instruction_file,
                            label_list=label_list,
                        )
                        wrapped = GenNERSampleWrapper(
                            id=instance_id,
                            dataset=dataset_dir.stem,
                            split=split_name,
                            label_list=label_list,
                            instance=instance,
                        )
                        output_file.fp.write(wrapped.model_dump_json() + "\n")
                        num_outputs += 1
                        prog.update()
                        if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                            logger.info(prog)
        logger.info(f"Number of samples in {output_file.path}: %d", num_outputs)


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


@app.command("compare_eval")
def compare_eval_results(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="convert_json_to_jsonl"),
        logging_home: str = typer.Option(default="output/GNER"),
        logging_file: str = typer.Option(default="compare_eval.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),
        # input
        input_file1: str = typer.Argument(default=...),
        input_file2: str = typer.Argument(default=...),
        # output
        output_file: str = typer.Argument(default=...),
        # option
        combine: bool = typer.Option(default="False"),
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
    file_opt1 = FileOption.from_path(
        path=input_file1,
        required=True,
    )
    file_opt2 = FileOption.from_path(
        path=input_file2,
        required=True,
    )
    args = CommonArguments(
        env=env,
    )

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(file_opt1) as input_file1,
        FileStreamer(file_opt2) as input_file2,
    ):
        logger.info("combine: %s", combine)
        logger.info("Loading input files: %s, %s", input_file1.path, input_file2.path)

        lines_1st = [x.replace("'", '"') for x in key_lines(key="'eval_runtime'", path=input_file1.path)]
        lines_2nd = [x.replace("'", '"') for x in key_lines(key="'eval_runtime'", path=input_file2.path)]

        metrics_1st = [GenNERMetrics.model_validate_json(x).calc() for x in lines_1st]
        metrics_2nd = [GenNERMetrics.model_validate_json(x).calc() for x in lines_2nd]

        df_1st = pd.DataFrame([x.model_dump() for x in metrics_1st]).set_index("epoch")
        df_2nd = pd.DataFrame([x.model_dump() for x in metrics_2nd]).set_index("epoch")

        logger.info("1st DataFrame:\n" + df_1st.to_string())
        logger.info("2nd DataFrame:\n" + df_2nd.to_string())

        if combine:
            df_diff = df_2nd.subtract(df_1st)
            combined_df = pd.concat([df_1st, df_2nd, df_diff], axis=1, keys=["1st", "2nd", "diff"])
            combined_df = combined_df[["1st", "2nd", "diff"]].stack(level=0)
            logger.info("Combined DataFrame:\n" + combined_df.to_string())
            combined_df.to_excel(output_file)
        else:
            with pd.ExcelWriter(output_file) as writer:
                df_1st.to_excel(writer, sheet_name='Sheet1', startrow=0)
                df_2nd.to_excel(writer, sheet_name='Sheet1', startrow=len(df_1st) + 2)
