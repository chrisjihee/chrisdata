from typing import Iterable

import typer
from qwikidata.datavalue import WikibaseEntityId, Time

from chrisbase.data import FileStreamer, MongoStreamer
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat, new_path
from chrisbase.util import mute_tqdm_cls
from chrisdata.wikidata import *

logger = logging.getLogger(__name__)
entity_cache: dict[str, Entity | None] = dict()
relation_cache: dict[str, Relation | None] = dict()


def get_entity(_id: str, reader: MongoStreamer) -> Entity | None:
    if _id not in entity_cache:
        row: dict | None = reader.table.find_one({'_id': _id})
        if not row:
            entity_cache[_id] = None
        else:
            row['id'] = row['_id']
            entity_cache[_id] = Entity.from_dict(row)
    return entity_cache[_id]


def get_relation(_id: str, reader: MongoStreamer) -> Relation | None:
    if _id not in relation_cache:
        row: dict | None = reader.table.find_one({'_id': _id})
        if not row:
            relation_cache[_id] = None
        else:
            row['id'] = row['_id']
            relation_cache[_id] = Relation.from_dict(row)
    return relation_cache[_id]


@dataclass
class TimeSensitiveEntity(DataClassJsonMixin):
    _id: str
    id: str


def extract_one(x: dict, args: IOArguments, reader: MongoStreamer) -> TimeSensitiveEntity | None:
    subject: WikidataUnit = WikidataUnit.from_dict(x)
    if subject.type == "item":
        claims = subject.claims
        subject: Entity = Entity.from_wikidata_item(subject)
        print("=" * 80)
        print(f"* subject: {subject.to_dict()}")
        for claim in claims:
            relation: Relation = get_relation(claim['property'], reader)
            if not relation:
                continue
            print(f"  + relation={relation}")
            datavalue: WikidataDatavalue = datavalue_dict_to_obj(claim['datavalue'])
            print(f"    - datavalue={datavalue}")
            if datavalue.datatype == "wikibase-entityid" and datavalue.value['entity-type'] == "item":
                object: Entity = get_entity(datavalue.value['id'], reader)
                if object:
                    print(f"    - object={object.to_dict()}")
            time_qualifiers = list()
            for qualifier in claim['qualifiers']:
                relation: Relation = get_relation(qualifier['property'], reader)
                datavalue: WikidataDatavalue = datavalue_dict_to_obj(qualifier['datavalue'])
                if relation and isinstance(datavalue, Time):
                    time_qualifiers.append({"relation": relation, "datavalue": datavalue})
            if time_qualifiers:
                print(f"    - time_qualifiers:")
                for qualifier in time_qualifiers:
                    print(f"      + {qualifier['relation']} = {qualifier['datavalue']}")
            print()
    return None


def extract_many(item: dict | Iterable[dict], args: IOArguments, reader: MongoStreamer, writer: MongoStreamer, item_is_batch: bool = True):
    batch = item if item_is_batch else [item]
    rows = [extract_one(x, args, reader) for x in batch]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


@app.command()
def extract(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="extract_wikidata"),
        logging_home: str = typer.Option(default="output/wikidata/extract"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=True),  # TODO: Replace with False
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),  # TODO: Replace with -1
        input_batch: int = typer.Option(default=1000),
        input_inter: int = typer.Option(default=5000),
        input_total: int = typer.Option(default=105485440),  # https://www.wikidata.org/wiki/Wikidata:Statistics  # TODO: Replace with (actual count)
        input_table_home: str = typer.Option(default="localhost:8800/Wikidata"),
        input_table_name: str = typer.Option(default="wikidata-20230911-all-parse-ko-en"),
        # output
        output_file_home: str = typer.Option(default="output/wikidata"),
        output_file_name: str = typer.Option(default="wikidata-20230911-all-extract.jsonl"),
        output_file_mode: str = typer.Option(default="w"),
        output_table_home: str = typer.Option(default="localhost:8800/Wikidata"),
        output_table_name: str = typer.Option(default="wikidata-20230911-all-extract"),
        output_table_reset: bool = typer.Option(default=True),
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
        # start=input_start if not debugging else 10751,
        start=input_start if not debugging else 0,
        limit=input_limit if not debugging else 2,
        batch=input_batch if not debugging else 1,
        inter=input_inter if not debugging else 1,
        total=input_total,
        table=TableOption(
            home=input_table_home,
            name=input_table_name,
            required=True,
        )
    )
    output_opt = OutputOption(
        file=FileOption(
            home=output_file_home,
            name=new_path(output_file_name, post=env.time_stamp),
            mode=output_file_mode,
            required=True,
        ),
        table=TableOption(
            home=output_table_home,
            name=output_table_name,
            reset=output_table_reset,
            required=True,
        )
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.table, "input.table is required"
    assert args.output.file, "output.file is required"
    assert args.output.table, "output.table is required"

    with(
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        MongoStreamer(args.input.table) as input_table,
        FileStreamer(args.output.file) as output_file,
        MongoStreamer(args.output.table) as output_table,
    ):
        # extract time-sensitive triples
        test_data = input_table.table.find({'_id': {'$in': ['Q50184', 'Q884']}})
        # input_data = args.input.ready_inputs(input_table, total=len(input_table))
        input_data = args.input.ready_inputs(test_data, total=len(input_table))
        logger.info(f"Extract from [{input_table.opt}] to [{output_table.opt}]")
        logger.info(f"- [input] total={args.input.total} | start={args.input.start} | limit={args.input.limit}"
                    f" | {type(input_data).__name__}={input_data.num_item}{f'x{args.input.batch}ea' if input_data.has_batch_items() else ''}")
        logger.info(f"- [output] table.reset={args.output.table.reset} | table.timeout={args.output.table.timeout}")
        with tqdm(total=input_data.num_item, unit="item", pre="=>", desc="extracting", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
            for item in input_data.items:
                extract_many(item=item, item_is_batch=input_data.has_batch_items(), args=args,
                             reader=input_table, writer=output_table)
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
