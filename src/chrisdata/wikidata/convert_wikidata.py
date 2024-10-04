from typing import Iterable

import httpx
import pandas as pd
import typer
from bs4 import BeautifulSoup
from dataclasses_json import DataClassJsonMixin
from qwikidata.datavalue import Time

from chrisbase.data import FileStreamer, MongoStreamer
from chrisbase.data import InputOption, OutputOption, FileOption, TableOption
from chrisbase.data import JobTimer, ProjectEnv, OptionData
from chrisbase.io import LoggingFormat, new_path, merge_dicts
from chrisbase.util import mute_tqdm_cls, grouped
from chrisdata.wikidata import *

logger = logging.getLogger(__name__)
entity_cache: dict[str, Entity | None] = dict()
relation_dict: dict[str, Relation | None] = dict()
list_of_all_properties: str = "https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all"


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
            row['id'] = row['_id']
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


def get_time(time_value: Time) -> str:
    units = time_value.get_parsed_datetime_dict()
    if time_value.value['precision'] <= 9:
        return f"{units['year']:04d}"
    elif time_value.value['precision'] == 10:
        return f"{units['year']:04d}-{units['month']:02d}"
    elif time_value.value['precision'] == 11:
        return f"{units['year']:04d}-{units['month']:02d}-{units['day']:02d}"
    else:
        raise ValueError(f"Unknown precision: {time_value.value['precision']}")


@dataclass
class TimeSensitiveEntity(DataClassJsonMixin):
    _id: str
    id: str


def convert_one(x: dict, args: IOArguments, reader: MongoStreamer) -> TimeSensitiveEntity | None:
    item: WikidataUnit = WikidataUnit.from_dict(reader.table.find_one({'_id': x}))
    claims = [x for x in item.claims if x['property'] in relation_dict]
    subject: Entity = Entity.from_wikidata_unit(item)
    print("=" * 80)
    print(f"* subject: {subject}")

    grouped_claims = {k: list(vs) for k, vs in grouped(claims, key=lambda i: i['property'])}
    for property, claim_group in grouped_claims.items():
        property_relation: Relation = relation_dict.get(property)
        num_claim = len(claim_group)
        print(f"  + {property_relation}: #object={num_claim}")
        for claim in claim_group:
            property_value: WikidataDatavalue = datavalue_dict_to_obj(claim['datavalue'])
            property_entity: Entity | None = None
            if property_value.datatype == "wikibase-entityid" and property_value.value['entity-type'] == "item":
                property_entity = get_entity(norm_wikidata_id(property_value.value['id']), reader)
            time_qualifiers = list()
            for qualifier in claim['qualifiers']:
                qualifier_relation: Relation = relation_dict.get(qualifier['property'])  # or f"{qualifier['property']}(필터링됨)"
                if not qualifier_relation:
                    continue
                qualifier_value: WikidataDatavalue = datavalue_dict_to_obj(qualifier['datavalue'])
                qualifier_entity: Entity | None = None
                if qualifier_value.datatype == "wikibase-entityid" and qualifier_value.value['entity-type'] == "item":
                    qualifier_entity = get_entity(norm_wikidata_id(qualifier_value.value['id']), reader)
                if isinstance(qualifier_value, Time):
                    time_qualifiers.append({"qualifier_relation": qualifier_relation, "qualifier_value": get_time(qualifier_value), "qualifier_entity": qualifier_entity})
                # else:
                #     time_qualifiers.append({"qualifier_relation": qualifier_relation, "qualifier_value": qualifier_value, "qualifier_entity": qualifier_entity})
            property_value = get_time(property_value) if isinstance(property_value, Time) else property_value
            print(f"    = [object] {property_entity or property_value}")
            if time_qualifiers:
                for qualifier in time_qualifiers:
                    print(f"      - {qualifier['qualifier_relation']} = {qualifier['qualifier_entity'] or qualifier['qualifier_value']}")
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
        exit(1)
        with tqdm(total=input_data.num_item, unit="item", pre="=>", desc="converting", unit_divisor=math.ceil(args.input.inter / args.input.batch)) as prog:
            for item in input_data.items:
                convert_many(item=item, args=args, reader=input_table, writer=output_table,
                             item_is_batch=input_data.has_batch_items())
                prog.update()
                if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                    logger.info(prog)
