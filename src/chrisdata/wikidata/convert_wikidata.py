from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd
import typer
from bs4 import BeautifulSoup
from flask import Flask, render_template
from more_itertools import ichunked
from pydantic import Field

from chrisbase.data import InputOption, OutputOption, FileOption, TableOption, FileStreamer
from chrisbase.data import JobTimer, ProjectEnv
from chrisbase.io import LoggingFormat, new_path, merge_dicts
from chrisbase.util import mute_tqdm_cls, grouped
from . import *
from ..wikipedia import WikipediaStat

logger = logging.getLogger(__name__)
relation_dict: dict[str, Relation | None] = dict()
wikipedia_stat: dict[str, WikipediaStat] = dict()
list_of_all_properties: str = "https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all"
datatype_orders: dict[str, int] = {
    "WI": 14, "WL": 13, "WP": 12, "WS": 11, "WF": 10,
    "TD": 9, "T": 8, "Q": 7, "GC": 6, "GS": 5, "M": 4,
    "MN": 3, "MT": 2, "S": 1,
    "U": -1, "CM": -2, "EI": -3, "ES": -4,
}


def property_order(k: str):
    return datatype_orders[relation_dict[k].datatype], relation_dict[k].property_count


class PageInfo(BaseModel):
    page: str
    num_entities: int
    first_entity: str
    last_entity: str


class ExtraOption(BaseModel):
    serve: bool = Field(default=False)
    export: bool = Field(default=False)
    processor: str | None = Field(default=None)
    serve_batch: int = Field(default=1000)
    min_property_count: int = Field(default=1000)
    black_property_datatypes: str = Field(default="CM|EI|ES|U")
    white_qualifier_relations: str = Field(default="P580|P582|P585")

    def black_property_datatype_list(self):
        return [x.strip() for x in self.black_property_datatypes.split("|")]

    def white_qualifier_relation_list(self):
        return [x.strip() for x in self.white_qualifier_relations.split("|")]


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


def convert_one(item_id: str, args: IOArguments, reader: MongoStreamer) -> SubjectStatements | None:
    item: WikidataUnit = WikidataUnit.from_dict(reader.table.find_one({'_id': item_id}))
    subject: Entity = Entity.from_wikidata_unit(item)
    if not subject.title1 or (subject.title1 not in wikipedia_stat) or wikipedia_stat[subject.title1].length < 1:
        return None
    if args.env.debugging:
        logger.info("*" * 80)
        logger.info(f" * {str([subject])[1:-1]}")

    num_statements: int = 0
    num_qualifiers: int = 0
    document_length: int = wikipedia_stat[subject.title1].length
    statements: list[Statement] = list()
    grouped_statements = {k: list(vs) for k, vs in grouped(item.claims, itemgetter='property') if k in relation_dict}
    statement_relations: list[Relation] = [
        relation_dict[k] for k in sorted(grouped_statements.keys(), key=property_order, reverse=True)
    ]
    args.option = ExtraOption.model_validate(args.option)
    white_qualifier_relation_list = args.option.white_qualifier_relation_list()
    for statement_relation in statement_relations:
        if args.env.debugging:
            logger.info(f"   + {str([statement_relation])[1:-1]}")
        statement_values: list[StatementValue] = list()
        for statement in grouped_statements[statement_relation.id]:
            statement_value: DataValue = datavalue_to_object(statement['datavalue'], reader)

            qualifiers: dict[str, str | None] = dict()
            grouped_qualifiers = {k: list(vs) for k, vs in grouped(statement['qualifiers'], itemgetter='property') if k in relation_dict}
            qualifier_relations: list[Relation] = [
                relation_dict[k] for k in sorted(grouped_qualifiers.keys(), key=property_order, reverse=True)
                if relation_dict[k].id in white_qualifier_relation_list
                # if relation_dict[k].datatype == "T"
            ]
            for qualifier_relation in qualifier_relations:
                qualifier_values: list[DataValue] = list()
                for qualifier in grouped_qualifiers[qualifier_relation.id]:
                    qualifier_value: DataValue = datavalue_to_object(qualifier['datavalue'], reader)
                    qualifier_values.append(qualifier_value)
                qualifiers[qualifier_relation.label2.replace(SP, US)] = ', '.join([i.string for i in qualifier_values])
                num_qualifiers += 1

            if args.env.debugging:
                logger.info(f"     - {str([StatementValue(value=statement_value, qualifiers=qualifiers)])[1:-1]}")
            statement_values.append(StatementValue(value=statement_value, qualifiers=qualifiers))
        statements.append(Statement(relation=statement_relation, values=statement_values))
        num_statements += 1

    if len(statements) > 0:
        if args.env.debugging:
            logger.info("-" * 80)
            logger.info(f"Subject: {str([subject])[1:-1]}")
            logger.info(f"Statements:")
            for x in statements:
                logger.info(f"- {str([x])[1:-1]}")
            logger.info("-" * 80)
            view_one(args, subject, statements)
        return SubjectStatements(
            subject=subject,
            statements=statements,
            num_statements=num_statements,
            num_qualifiers=num_qualifiers,
            document_length=document_length,
        )
    return None


def view_one(args: IOArguments, subject: Entity, statements: list[Statement]):
    server = Flask("wikidata_browser", template_folder=args.env.working_dir / "templates")

    @server.route("/")
    def index():
        return render_template("entity_detail.html", subject=subject, statement_list=statements)

    server.run(host="localhost", port=7321, debug=False)


def convert_many(item: str | Iterable[str], args: IOArguments, reader: MongoStreamer, writer: MongoStreamer, item_is_batch: bool = True):
    inputs = item if item_is_batch else [item]
    outputs = [convert_one(i, args, reader) for i in inputs]
    outputs = {v.subject.title1: v for v in outputs if v}
    records = [merge_dicts({"_id": k}, v.model_dump()) for k, v in outputs.items() if v]
    if len(records) > 0:
        writer.table.insert_many(records)


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
        input_start: int = typer.Option(default=0),
        input_limit: int = typer.Option(default=-1),
        input_batch: int = typer.Option(default=500),
        input_inter: int = typer.Option(default=500),
        input_file_path: str = typer.Option(default="input/wikidata/wikidata-20240916-korean-full.txt"),
        input_prop_path: str = typer.Option(default="input/wikidata/wikidata-properties.jsonl"),
        input_stat_path: str = typer.Option(default="input/wikipedia/kowiki-20230701-all-titles-in-ns0-stat.jsonl"),
        input_table_path: str = typer.Option(default="localhost:8800/wikidata/wikidata-20240916-parse"),
        input_table_timeout: int = typer.Option(default=3600),
        # output
        output_file_path: str = typer.Option(default="output/wikidata/wikidata-20240916-convert.jsonl"),
        output_file_mode: str = typer.Option(default="w"),
        output_table_path: str = typer.Option(default="localhost:8800/wikidata/wikidata-20240916-convert"),
        output_table_reset: bool = typer.Option(default=True),
        # option
        serve: bool = typer.Option(default=True),
        export: bool = typer.Option(default=True),
        processor: str = typer.Option(default="convert_many"),
        serve_batch: int = typer.Option(default=1000),
        min_property_count: int = typer.Option(default=1000),
        black_property_datatypes: str = typer.Option(default="CM|EI|ES|U"),
        white_qualifier_relations: str = typer.Option(default="P580|P582|P585"),
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
        file=FileOption.from_path(
            path=input_file_path,
            required=True,
        ),
        table=TableOption.from_path(
            path=input_table_path,
            timeout=input_table_timeout * 1000,
            required=True,
        )
    )
    output_opt = OutputOption(
        file=FileOption.from_path(
            path=output_file_path,
            name=new_path(output_file_path, post=env.time_stamp).name,
            mode=output_file_mode,
            required=True,
        ),
        table=TableOption.from_path(
            path=output_table_path,
            reset=output_table_reset,
            required=True,
        )
    )
    extra_opt = ExtraOption(
        serve=serve,
        export=export,
        processor=processor,
        serve_batch=serve_batch,
        min_property_count=min_property_count,
        black_property_datatypes=black_property_datatypes,
        white_qualifier_relations=white_qualifier_relations,
    )
    args = IOArguments(
        env=env,
        input=input_opt,
        output=output_opt,
        option=extra_opt,
    )
    tqdm = mute_tqdm_cls()
    assert args.input.file, "input.file is required"
    assert args.input.table, "input.table is required"
    assert args.output.file, "output.file is required"
    assert args.output.table, "output.table is required"

    with (
        JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='),
        FileStreamer(FileOption.from_path(input_stat_path, required=True)) as stat_file,
        FileStreamer(FileOption.from_path(input_prop_path, required=False)) as prop_file,
        FileStreamer(args.input.file) as input_file, MongoStreamer(args.input.table) as input_table,
        FileStreamer(args.output.file) as output_file, MongoStreamer(args.output.table) as output_table,
    ):
        global wikipedia_stat, relation_dict
        for i in stat_file:
            v = WikipediaStat.model_validate_json(i)
            wikipedia_stat[v.id] = v
        logger.info(f"Load Wikipedia {len(wikipedia_stat)} statistics from {stat_file.path}")
        if prop_file.fp:
            total_properties: pd.DataFrame = pd.read_json(prop_file.path, orient='records', lines=True)
            source = prop_file.path
        else:
            total_properties: pd.DataFrame = download_wikidata_properties()
            total_properties.to_json(prop_file.path, orient='records', lines=True)
            source = list_of_all_properties
        logger.info(f"Load Wikidata {total_properties.shape[0]} properties from {source}")
        valid_properties = total_properties[
            ~total_properties['datatype'].isin(extra_opt.black_property_datatype_list())
            & ((total_properties['property_count'] >= extra_opt.min_property_count)
               | (total_properties['qualifier_count'] >= extra_opt.min_property_count))
            ]
        logger.info(f"Keep Wikidata {valid_properties.shape[0]} properties by options: min_property_count={extra_opt.min_property_count}, black_property_datatypes={extra_opt.black_property_datatypes}")
        for p in valid_properties.to_dict(orient='records'):
            row = input_table.table.find_one({'_id': norm_wikidata_id(p['ID'])})
            relation_dict[p['ID']] = Relation.model_validate(merge_dicts(row, {
                'datatype': p['datatype'],
                'property_count': p['property_count'],
                'qualifier_count': p['qualifier_count'],
                'reference_count': p['reference_count'],
            }))
        logger.info(f"Make relation_dict for Wikidata {len(relation_dict)} properties using {input_table.opt}")

        # convert time-sensitive triples
        input_data = args.input.ready_inputs(input_file, total=len(input_file))
        logger.info(f"Convert from [{input_file.opt}, {input_table.opt}]")
        logger.info(f"          to [{output_file.opt}, {output_table.opt}]")
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

        if extra_opt.export:
            with tqdm(total=len(output_table), unit="row", pre="=>", desc="exporting", unit_divisor=args.input.inter * 100) as prog:
                for row in output_table:
                    output_file.fp.write(SubjectInfo.model_validate(row).model_dump_json() + '\n')
                    prog.update()
                    if prog.n == prog.total or prog.n % prog.unit_divisor == 0:
                        logger.info(prog)
                logger.info(f"Export {prog.n}/{args.input.total} rows to [{output_file.opt}]")

        if extra_opt.serve:
            entity_list: list[SubjectInfo] = list()
            entity_details: dict[str, SubjectStatements] = dict()
            for row in output_table:
                info: SubjectInfo = SubjectInfo.model_validate(row)
                detail: SubjectStatements = SubjectStatements.model_validate(row)
                entity_list.append(info)
                entity_details[info.subject.id] = detail

            logger.info(f"Load {len(entity_list)} entities from [{output_table.opt}]")
            server = Flask("wikidata_browser", template_folder=args.env.working_dir / "templates")
            page_dict: dict[int, list[SubjectInfo]] = {
                i: list(xs)
                for i, xs in enumerate(ichunked(entity_list, extra_opt.serve_batch), start=1)
            }
            page_list: list[PageInfo] = [
                PageInfo(page=f"{i:04d}", num_entities=len(xs), first_entity=xs[0].subject.title1, last_entity=xs[-1].subject.title1)
                for i, xs in page_dict.items()
            ]
            logger.info(f"Chunk {len(entity_list)} entities into {extra_opt.serve_batch} entities * {len(page_dict)} pages")

            @server.route("/")
            def index():
                return render_template("page_list.html", page_list=page_list)
                # return redirect(url_for('render_entity_detail', sub='Q1.html'))

            @server.route("/<string:page>.html")
            def render_entity_list(page: str):
                try:
                    page = int(page)
                except ValueError:
                    return f"Page Error (page={page})", 404
                if page in page_dict:
                    return render_template("entity_list.html", entity_list=page_dict[page])
                else:
                    return "Not Found", 404

            @server.route("/entity/<path:sub>")
            def render_entity_detail(sub: str):
                id = Path(sub).stem
                entity: SubjectStatements | None = entity_details.get(id)
                if entity:
                    return render_template("entity_detail.html", subject=entity.subject, statement_list=entity.statements)
                else:
                    return "Not Found", 404

            server.run(host="localhost", port=7321, debug=False)
