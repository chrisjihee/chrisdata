import bz2
import logging
import os
from dataclasses import dataclass, field

import pandas as pd
import typer
from chrisbase.data import AppTyper, ProjectEnv, CommonArguments, JobTimer
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe
from rdflib import Graph
from wikibaseintegrator import WikibaseIntegrator

logger = logging.getLogger(__name__)
app = AppTyper()


def ntriples_to_dicts(ntriples_text):
    # N-Triples 파싱
    g = Graph()
    g.parse(data=ntriples_text, format="nt")

    # RDF 그래프를 JSON으로 변환
    rdf_data = []
    for subject, predicate, obj in g:
        rdf_data.append({
            "subject": str(subject),
            "predicate": str(predicate),
            "object": str(obj),
        })

    return rdf_data


@dataclass
class ProgramArguments(CommonArguments):
    other: str | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.time, data_prefix="time"),
            to_dataframe(columns=columns, raw={"other": self.other}),
        ]).reset_index(drop=True)


@app.command()
def parse(
        # env
        project: str = typer.Option(default="WiseData"),
        job_name: str = typer.Option(default="parse_wikidata"),
        output_home: str = typer.Option(default="output-parse_wikidata"),
        max_workers: int = typer.Option(default=os.cpu_count()),
        debugging: bool = typer.Option(default=False),
):
    args = ProgramArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name,
            debugging=debugging,
            output_home=output_home,
            msg_level=logging.DEBUG if debugging else logging.INFO,
            msg_format=LoggingFormat.DEBUG_48 if debugging else LoggingFormat.CHECK_36,
            max_workers=1 if debugging else max(max_workers, 1),
        ),
        other="other",
    )

    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        wbi = WikibaseIntegrator()
        p = wbi.property.get('P1376')
        logger.info(p.labels.values['en'])
        logger.info(p.labels.values['ko'])
        logger.info(p.descriptions.values['en'])
        logger.info(p.descriptions.values['ko'])
        num_line = 20
        with bz2.BZ2File("/fed/Wikidata/latest-truthy-nt-bz2/latest-truthy.nt.bz2") as f:
            n = 0
            for line in f:
                n += 1
                if n > num_line:
                    break
                logger.info(ntriples_to_dicts(line))


if __name__ == "__main__":
    app()
