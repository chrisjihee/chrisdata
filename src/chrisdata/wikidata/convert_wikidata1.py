from pathlib import Path

import jsonlines
from dataclasses_json import DataClassJsonMixin

from chrisbase.io import num_lines, LoggingFormat, setup_unit_logger
from chrisbase.util import mute_tqdm_cls
from . import *

logger = logging.getLogger(__name__)
setup_unit_logger(fmt=LoggingFormat.CHECK_12)


@dataclass
class WikidataFreebaseID(DataClassJsonMixin):
    wikidata_id: str
    freebase_id: str
    label1: str
    label2: str
    title1: str
    title2: str
    descr1: str
    descr2: str


input_file = Path('input/wikimedia/wikidata-20230920-parse-kowiki.jsonl')
output_file = Path('input/wikimedia/wikidata-20230920-freebase.jsonl')
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
        unit = WikidataUnit.from_dict(x)
        if unit.type == "item":
            for claim in unit.claims:
                if claim['property'] == "P646" and claim['datavalue']['type'] == "string":
                    freebase_id = claim['datavalue']['value']
                    writer.writelines([
                        WikidataFreebaseID(
                            wikidata_id=unit._id,
                            freebase_id=freebase_id,
                            label1=unit.label1,
                            label2=unit.label2,
                            title1=unit.title1,
                            title2=unit.title2,
                            descr1=unit.descr1,
                            descr2=unit.descr2,
                        ).to_json(ensure_ascii=False),
                        '\n'
                    ])
