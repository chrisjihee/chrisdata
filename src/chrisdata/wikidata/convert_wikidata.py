from typing import Iterable

import httpx
import pandas as pd
import typer
from bs4 import BeautifulSoup
from dataclasses_json import DataClassJsonMixin
from qwikidata.datavalue import Time, Quantity, WikibaseEntityId, String, MonolingualText

from chrisbase.data import FileStreamer, MongoStreamer
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv, OptionData
from chrisbase.io import LoggingFormat, new_path, merge_dicts
from chrisbase.util import mute_tqdm_cls, grouped, SP, CM, US
from chrisdata.wikidata import *

logger = logging.getLogger(__name__)
entity_cache: dict[str, Entity | None] = dict()
relation_dict: dict[str, Relation | None] = dict()
list_of_all_properties: str = "https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all"
QUANTITY_UNIT_PATTERN = re.compile(
    r"""
    http://www.wikidata.org/entity/(?P<id>[A-Z]\d+)
    """,
    re.VERBOSE,
)
datatype_orders: dict[str, int] = {
    "WI": 14,
    "WL": 13,
    "WP": 12,
    "WS": 11,
    "WF": 10,
    "TD": 9,
    "T": 8,
    "Q": 7,
    "GC": 6,
    "GS": 5,
    "M": 4,
    "MN": 3,
    "MT": 2,
    "S": 1,
    "U": -1,
    "CM": -2,
    "EI": -3,
    "ES": -4,
}


def property_order(k: str):
    return datatype_orders[relation_dict[k].datatype], relation_dict[k].property_count


@dataclass
class ExtraOption(OptionData):
    min_property_count: int = field(default=1000)
    black_property_datatypes: str = field(default="CM, EI, ES, U")


def download_wikidata_properties() -> pd.DataFrame:
    with httpx.Client(
            timeout=httpx.Timeout(timeout=120.0)
    ) as cli:
        response = cli.get(list_of_all_properties)
        # from pathlib import Path
        # Path("test.html").write_text(response.text)
        soup = BeautifulSoup(response.text, "html.parser")
        columns = [
            ([sup.decompose() for sup in th.select("sup")], th.text.strip())[-1]
            for th in soup.select_one("table.wikitable tr").select("th")
        ]
        data = []
        for tr in soup.select("table.wikitable tr")[1:]:
            body_values = [' | '.join(td.stripped_strings) for td in tr.select("td")]
            data.append(body_values)
        data = pd.DataFrame(data, columns=columns)
        data['property_count'] = data['Counts'].str.extract(r'([0-9,]+) *M')[0].str.replace(',', '').astype(float).fillna(0).astype(int)
        data['qualifier_count'] = data['Counts'].str.extract(r'([0-9,]+) *Q')[0].str.replace(',', '').astype(float).fillna(0).astype(int)
        data['reference_count'] = data['Counts'].str.extract(r'([0-9,]+) *R')[0].str.replace(',', '').astype(float).fillna(0).astype(int)
        data = data.drop(columns=['Counts']).rename(columns={'Data type': 'datatype'})
        return data


def get_entity(_id: str, reader: MongoStreamer) -> Entity | None:
    if _id not in entity_cache:
        row: dict | None = reader.table.find_one({'_id': _id})
        if not row:
            entity_cache[_id] = None
        else:
            entity_cache[_id] = Entity.from_dict(row)
    return entity_cache[_id]


# def get_relation(_id: str, reader: MongoStreamer) -> Relation | None:
#     if _id not in relation_dict:
#         row: dict | None = reader.table.find_one({'_id': _id})
#         if not row:
#             relation_dict[_id] = None
#         else:
#             row['id'] = row['_id']
#             relation_dict[_id] = Relation.from_dict(row)
#     return relation_dict[_id]


def get_wikidata_entity(datavalue: WikibaseEntityId, reader: MongoStreamer) -> str:
    entity: Entity | None = get_entity(norm_wikidata_id(datavalue.value['id']), reader)
    if not entity:
        return f"{datavalue.value['id']}"
    else:
        return (entity.title1 or entity.title2) or (entity.label1 or entity.label2) or entity.id
        # return f"Entity(title={entity.title1 or entity.title2}, label={entity.label1 or entity.label2}, id={entity.id})"


def get_quantity(datavalue: Quantity, reader: MongoStreamer) -> str:
    if datavalue.value['unit'] == '1':
        return f"{datavalue.value['amount']}"
    match = QUANTITY_UNIT_PATTERN.fullmatch(datavalue.value['unit'])
    if not match:
        return f"{datavalue.value['amount']} {datavalue.value['unit']}"
    else:
        unit: Entity | None = get_entity(norm_wikidata_id(match.group('id')), reader)
        return f"{datavalue.value['amount']} {unit.label2 or unit.label1 or unit.id}"


def get_time(datavalue: Time) -> str:
    parts = datavalue.get_parsed_datetime_dict()
    if datavalue.value['precision'] <= 9:
        return f"{parts['year']:04d}"
    elif datavalue.value['precision'] == 10:
        return f"{parts['year']:04d}-{parts['month']:02d}"
    elif datavalue.value['precision'] == 11:
        return f"{parts['year']:04d}-{parts['month']:02d}-{parts['day']:02d}"
    else:
        raise ValueError(f"Unknown precision: {datavalue.value['precision']}")


def get_monolingual_text(datavalue: MonolingualText) -> str:
    return f"{datavalue.value['text']}({datavalue.value['language']})"


def datavalue_dict_to_str(datavalue: dict, reader: MongoStreamer) -> str:
    obj: WikidataDatavalue = datavalue_dict_to_obj(datavalue)
    if isinstance(obj, WikibaseEntityId):
        return get_wikidata_entity(obj, reader)
    elif isinstance(obj, Quantity):
        return get_quantity(obj, reader)
    elif isinstance(obj, Time):
        return get_time(obj)
    elif isinstance(obj, MonolingualText):
        return get_monolingual_text(obj)
    elif isinstance(obj, String):
        return obj.value
    else:
        return str(obj)


@dataclass
class TimeSensitiveEntity(DataClassJsonMixin):
    _id: str
    id: str


def convert_one(x: dict, args: IOArguments, reader: MongoStreamer) -> TimeSensitiveEntity | None:
    item: WikidataUnit = WikidataUnit.from_dict(reader.table.find_one({'_id': x}))
    subject: Entity = Entity.from_wikidata_unit(item)
    print("=" * 80)
    print(f"* subject: {subject}")

    statements = {k: list(vs) for k, vs in grouped(item.claims, itemgetter='property') if k in relation_dict}
    statement_relations = [
        relation_dict[k] for k in sorted(statements.keys(), key=property_order, reverse=True)
    ]
    for statement_relation in statement_relations:
        print(statement_relation.id, statement_relation.label1, statement_relation.label2, statement_relation.datatype, statement_relation.property_count, len(statements[statement_relation.id]))
        for statement in statements[statement_relation.id]:
            statement_value_str = datavalue_dict_to_str(statement['datavalue'], reader)
            statement_value: WikidataDatavalue = datavalue_dict_to_obj(statement['datavalue'])

            qualifiers = {k: list(vs) for k, vs in grouped(statement['qualifiers'], itemgetter='property') if k in relation_dict}
            qualifier_relations = [
                relation_dict[k] for k in sorted(qualifiers.keys(), key=property_order, reverse=True)
                if relation_dict[k].datatype == "T"
            ]
            time_qualifiers = list()
            for qualifier_relation in qualifier_relations:
                qualifier_values = [
                    datavalue_dict_to_str(qualifier['datavalue'], reader)
                    for qualifier in qualifiers[qualifier_relation.id]
                ]
                time_qualifiers.append(f"{qualifier_relation.label2.replace(SP, US)}={'|'.join(qualifier_values)}")
            print(f"= {statement_value_str} ({type(statement_value).__name__}){f' ({(CM + SP).join(time_qualifiers)})' if time_qualifiers else ''}")
        print()


def convert_many(item: dict | Iterable[dict], args: IOArguments, reader: MongoStreamer, writer: MongoStreamer, item_is_batch: bool = True):
    batch = item if item_is_batch else [item]
    rows = [convert_one(x, args, reader) for x in batch]
    rows = [row.to_dict() for row in rows if row]
    if len(rows) > 0:
        writer.table.insert_many(rows)


@app.command()
def convert(
        # env
        project: str = typer.Option(default="chrisdata"),
        job_name: str = typer.Option(default="convert_wikidata"),
        logging_home: str = typer.Option(default="output/wikidata/convert"),
        logging_file: str = typer.Option(default="logging.out"),
        max_workers: int = typer.Option(default=1),
        debugging: bool = typer.Option(default=False),  # TODO: Replace with False
        # input
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),  # TODO: Replace with -1
        input_batch: int = typer.Option(default=100),  # TODO: Replace with 100
        input_inter: int = typer.Option(default=100),  # TODO: Replace with 10000
        input_file_home: str = typer.Option(default="input/wikidata"),
        input_file_name: str = typer.Option(default="wikidata-20240916-korean.txt"),
        input_prop_name: str = typer.Option(default="wikidata-properties.jsonl"),
        input_table_home: str = typer.Option(default="localhost:8800/wikidata"),
        input_table_name: str = typer.Option(default="wikidata-20240916-parse"),
        input_table_timeout: int = typer.Option(default=3600),
        # output
        output_file_home: str = typer.Option(default="output/wikidata"),
        output_file_name: str = typer.Option(default="wikidata-20240916-convert.jsonl"),
        output_file_mode: str = typer.Option(default="w"),
        output_table_home: str = typer.Option(default="localhost:8800/wikidata"),
        output_table_name: str = typer.Option(default="wikidata-20240916-convert"),
        output_table_reset: bool = typer.Option(default=True),
        # option
        min_property_count: int = typer.Option(default=1000),
        black_property_datatypes: str = typer.Option(default="CM, EI, ES, U"),
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
        start=input_start if not debugging else 0,
        limit=input_limit if not debugging else 2,
        batch=input_batch if not debugging else 1,
        inter=input_inter if not debugging else 1,
        file=FileOption(
            home=input_file_home,
            name=input_file_name,
            required=True,
        ),
        table=TableOption(
            home=input_table_home,
            name=input_table_name,
            timeout=input_table_timeout * 1000,
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
    extra_opt = ExtraOption(
        min_property_count=min_property_count,
        black_property_datatypes=black_property_datatypes,
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
        option=extra_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.table, "input.table is required"
    assert args.output.file, "output.file is required"
    assert args.output.table, "output.table is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(FileOption(home=input_file_home, name=input_prop_name)) as prop_file,
        FileStreamer(args.input.file) as input_file, MongoStreamer(args.input.table) as input_table,
        FileStreamer(args.output.file) as output_file, MongoStreamer(args.output.table) as output_table,
    ):
        if prop_file.fp:
            total_properties: pd.DataFrame = pd.read_json(prop_file.path, orient='records', lines=True)
            source = prop_file.path
        else:
            total_properties: pd.DataFrame = download_wikidata_properties()
            total_properties.to_json(prop_file.path, orient='records', lines=True)
            source = list_of_all_properties
        logger.info(f"Load Wikidata {total_properties.shape[0]} properties from {source}")
        valid_properties = total_properties[
            ~total_properties['datatype'].isin([x.strip() for x in args.option.black_property_datatypes.split(",")])
            & ((total_properties['property_count'] >= args.option.min_property_count)
               | (total_properties['qualifier_count'] >= args.option.min_property_count))
            ]
        logger.info(f"Keep Wikidata {valid_properties.shape[0]} properties by options: min_property_count={args.option.min_property_count}, black_property_datatypes={args.option.black_property_datatypes}")
        for p in valid_properties.to_dict(orient='records'):
            row = input_table.table.find_one({'_id': norm_wikidata_id(p['ID'])})
            relation_dict[p['ID']] = Relation.from_dict(merge_dicts(row, {
                'datatype': p['datatype'],
                'property_count': p['property_count'],
                'qualifier_count': p['qualifier_count'],
                'reference_count': p['reference_count'],
            }))
        logger.info(f"Make relation_dict for Wikidata {len(relation_dict)} properties using {input_table.opt}")
        # for x in relation_dict.items():
        #     print(x)
        # exit(1)

        # convert time-sensitive triples
        # test_data = input_table.table.find({'_id': {'$in': [norm_wikidata_id('Q50184'), norm_wikidata_id('Q884')]}})
        # input_data = args.input.ready_inputs(test_data, total=len(input_table))
        # input_data = args.input.ready_inputs(input_table, total=len(input_table))
        input_data = args.input.ready_inputs(input_file, total=len(input_file))
        logger.info(f"Convert from [{input_file.opt}, {input_table.opt}] to [{output_file.opt}, {output_table.opt}]")
        logger.info(f"- [input] total={args.input.total} | start={args.input.start} | limit={args.input.limit}"
                    f" | {type(input_data).__name__}={input_data.num_item}{f'x{args.input.batch}ea' if input_data.has_batch_items() else ''}")
        logger.info(f"- [input] table.timeout={args.input.table.timeout}")
        logger.info(f"- [output] file.reset={args.output.file.reset} | file.mode={args.output.file.mode}")
        logger.info(f"- [output] table.reset={args.output.table.reset} | table.timeout={args.output.table.timeout}")
        with tqdm(total=input_data.num_item, unit="item", pre="=>", desc="converting", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
            for item in input_data.items:
                convert_many(item=item, args=args, reader=input_table, writer=output_table,
                             item_is_batch=input_data.has_batch_items())
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
