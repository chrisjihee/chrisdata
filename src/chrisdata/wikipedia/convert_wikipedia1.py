import logging
import re
from pathlib import Path

import jsonlines

from chrisbase.io import num_lines, LoggingFormat, setup_unit_logger
from chrisbase.util import mute_tqdm_cls
from . import *

logger = logging.getLogger(__name__)
setup_unit_logger(fmt=LoggingFormat.CHECK_12)


@dataclass
class WikipediaDefinition(DataClassJsonMixin):
    page_id: int
    query: str
    title: str
    definition: str


input_file = Path('input/wikimedia/wikipedia-20230920-crawl-kowiki.jsonl')
output_file = Path('input/wikimedia/wikipedia-20230920-definition.jsonl')
tqdm = mute_tqdm_cls()

logger.info(f"Input file: {input_file}")
logger.info(f"Output file: {output_file}")

with jsonlines.open(input_file) as reader, output_file.open("w") as writer:
    progress, interval = (
        tqdm(reader, total=num_lines(input_file), unit="line", pre="*", desc="converting"),
        10_000,
    )
    for i, x in enumerate(progress):
        if i > 0 and i % interval == 0:
            logger.info(progress)
        doc: WikipediaProcessResult = WikipediaProcessResult.from_dict(x)
        doc.title = doc.title.strip() if doc.title else ""
        if doc.page_id and doc.title and doc.section_list:
            definition = doc.section_list[0][-1]
            definition = re.sub(r"\s+", " ", definition)
            definition = re.sub("다\\..*?$", "다.", definition)
            if not definition.endswith("다른 뜻은 다음과 같다."):
                writer.writelines([
                    WikipediaDefinition(
                        page_id=doc.page_id,
                        query=doc.query,
                        title=doc.title,
                        definition=definition,
                    ).to_json(ensure_ascii=False),
                    '\n'
                ])
    logger.info(progress)