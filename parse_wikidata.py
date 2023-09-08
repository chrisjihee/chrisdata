import logging
import os
from dataclasses import dataclass, field

import pandas as pd
import typer
from qwikidata.sparql import return_sparql_query_results
from wikibaseintegrator import WikibaseIntegrator

from chrisbase.data import AppTyper, ProjectEnv, CommonArguments, JobTimer
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe

logger = logging.getLogger(__name__)
app = AppTyper()


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
        pass
    wbi = WikibaseIntegrator()
    p = wbi.property.get('P1376')
    logger.info(p.labels.values['en'])
    logger.info(p.labels.values['ko'])
    logger.info(p.descriptions.values['en'])
    logger.info(p.descriptions.values['ko'])
    sparql_query = """
    SELECT (COUNT(?item) AS ?count)
    WHERE {
            ?item wdt:P31/wdt:P279* wd:Q5 .
    }
    """
    res = return_sparql_query_results(sparql_query)
    logger.info(res)


if __name__ == "__main__":
    app()
