import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import typer
from chrisbase.data import AppTyper, JobTimer, ProjectEnv, OptionData, CommonArguments
from chrisbase.io import LoggingFormat
from chrisbase.util import to_dataframe
from qwikidata.claim import WikidataClaim
from qwikidata.entity import WikidataItem, WikidataProperty
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.typedefs import LanguageCode

logger = logging.getLogger(__name__)
app = AppTyper()


class WikidataItemEx(WikidataItem):
    def get_wiki_title(self, lang) -> str:
        """Get english language wikipedia page title."""
        wikiname = f"{str(lang).lower()}wiki"
        if (
                isinstance(self._entity_dict["sitelinks"], dict)
                and wikiname in self._entity_dict["sitelinks"]
        ):
            return self._entity_dict["sitelinks"][wikiname]["title"]
        else:
            return ""

    def get_truthy_claims(self) -> list[dict]:
        claim_groups = self.get_truthy_claim_groups()
        claims = list()
        for claim_group in claim_groups.values():
            for claim in claim_group:
                claim: WikidataClaim = claim
                if claim.mainsnak.snaktype == "value" and claim.mainsnak.datavalue is not None:
                    claims.append({"property": claim.mainsnak.property_id, "datavalue": claim.mainsnak.datavalue._datavalue_dict})
        return claims


class WikidataPropertyEx(WikidataProperty):
    def get_truthy_claims(self) -> list[dict]:
        claim_groups = self.get_truthy_claim_groups()
        claims = list()
        for claim_group in claim_groups.values():
            for claim in claim_group:
                claim: WikidataClaim = claim
                if claim.mainsnak.snaktype == "value" and claim.mainsnak.datavalue is not None:
                    claims.append({"property": claim.mainsnak.property_id, "datavalue": claim.mainsnak.datavalue._datavalue_dict})
        return claims


@dataclass
class DataOption(OptionData):
    home: str | Path = field()
    name: str | Path = field()
    lang1: str = field(default="ko")
    lang2: str = field(default="en")
    limit: int = field(default=-1)
    from_scratch: bool = field(default=False)

    def __post_init__(self):
        self.home = Path(self.home)
        self.name = Path(self.name)
        self.lang1_code = LanguageCode(self.lang1)
        self.lang2_code = LanguageCode(self.lang2)


@dataclass
class ProgramArguments(CommonArguments):
    data: DataOption | None = field(default=None)
    other: str | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()

    def dataframe(self, columns=None) -> pd.DataFrame:
        if not columns:
            columns = [self.data_type, "value"]
        return pd.concat([
            to_dataframe(columns=columns, raw=self.env, data_prefix="env"),
            to_dataframe(columns=columns, raw=self.data, data_prefix="data") if self.data else None,
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
        # data
        input_home: str = typer.Option(default="input/Wikidata"),
        input_name: str = typer.Option(default="latest-all.json.bz2"),
        input_limit: int = typer.Option(default=-1),
        input_lang1: str = typer.Option(default="ko"),
        input_lang2: str = typer.Option(default="en"),
        from_scratch: bool = typer.Option(default=False),
        # etc
        use_tqdm: bool = typer.Option(default=True),
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
        data=DataOption(
            home=input_home,
            name=input_name,
            lang1=input_lang1,
            lang2=input_lang2,
            limit=input_limit,
            from_scratch=from_scratch,
        ),
        other="other",
    )

    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", args=args, rt=1, rb=1, rc='='):
        # from wikibaseintegrator import WikibaseIntegrator
        # wbi = WikibaseIntegrator()
        # p = wbi.property.get('P1376')
        # logger.info(p.labels.values['en'])
        # logger.info(p.labels.values['ko'])
        # logger.info(p.descriptions.values['en'])
        # logger.info(p.descriptions.values['ko'])

        wikidata = WikidataJsonDump(str(args.data.home / args.data.name))
        for ii, entity_dict in enumerate(wikidata):
            if 0 < args.data.limit < ii + 1:
                break
            if entity_dict['type'] == "item" and entity_dict['ns'] == 0:
                continue
                item = WikidataItemEx(entity_dict)
                logger.info(item)
                logger.info(f"- ii: {ii}")
                logger.info(f"- id: {item.entity_id}")
                logger.info(f"- ns: {entity_dict['ns']}")
                logger.info(f"- type: {item.entity_type}")
                logger.info(f"- time: {entity_dict['modified']}")
                logger.info(f"- label1: {item.get_label(args.data.lang1_code)}")
                logger.info(f"- label2: {item.get_label(args.data.lang2_code)}")
                logger.info(f"- descr1: {item.get_description(args.data.lang1_code)}")
                logger.info(f"- descr2: {item.get_description(args.data.lang2_code)}")
                logger.info(f"- alias1: {item.get_aliases(args.data.lang1_code)}")
                logger.info(f"- alias2: {item.get_aliases(args.data.lang2_code)}")
                logger.info(f"- title1: {item.get_wiki_title(args.data.lang1)}")
                logger.info(f"- title2: {item.get_wiki_title(args.data.lang2)}")
                claims = item.get_truthy_claims()
                logger.info(f"- claims({len(claims)}): {claims}")
                logger.info("----")
                for k, v in entity_dict.items():
                    if k in ("claims",):
                        continue
                    if k in ("labels", "descriptions"):
                        entity_dict[k] = [vv for kk, vv in v.items() if kk in ("en", "ko")]
                    if k in ("aliases",):
                        entity_dict[k] = [vvv for kk, vv in v.items() if kk in ("en", "ko") for vvv in vv]
                    if k in ("sitelinks",):
                        entity_dict[k] = [vv for kk, vv in v.items() if kk in ("enwiki", "kowiki")]
                    logger.info("- {}: {}".format(k, entity_dict[k]))
                logger.info("====")
            elif entity_dict['type'] == "property":
                prop = WikidataPropertyEx(entity_dict)
                logger.info(prop)
                logger.info(f"- ii: {ii}")
                logger.info(f"- id: {prop.entity_id}")
                logger.info(f"- ns: {entity_dict['ns']}")
                logger.info(f"- type: {prop.entity_type}")
                logger.info(f"- time: {entity_dict['modified']}")
                logger.info(f"- label1: {prop.get_label(args.data.lang1_code)}")
                logger.info(f"- label2: {prop.get_label(args.data.lang2_code)}")
                logger.info(f"- descr1: {prop.get_description(args.data.lang1_code)}")
                logger.info(f"- descr2: {prop.get_description(args.data.lang2_code)}")
                logger.info(f"- alias1: {prop.get_aliases(args.data.lang1_code)}")
                logger.info(f"- alias2: {prop.get_aliases(args.data.lang2_code)}")
                claims = prop.get_truthy_claims()
                logger.info(f"- claims({len(claims)}): {claims}")
                logger.info("----")
                for k, v in entity_dict.items():
                    if k in ("claims",):
                        continue
                    if k in ("labels", "descriptions"):
                        entity_dict[k] = [vv for kk, vv in v.items() if kk in ("en", "ko")]
                    if k in ("aliases",):
                        entity_dict[k] = [vvv for kk, vv in v.items() if kk in ("en", "ko") for vvv in vv]
                    if k in ("sitelinks",):
                        entity_dict[k] = [vv for kk, vv in v.items() if kk in ("enwiki", "kowiki")]
                    logger.info("- {}: {}".format(k, entity_dict[k]))
                logger.info("====")
            else:
                logger.info(f"BREAK ii = {ii}")
                logger.info(f"entity_dict['type'] = {entity_dict['type']}")
                logger.info(entity_dict)
                exit(3)
        logger.info(f"FINAL ii = {ii}")


if __name__ == "__main__":
    app()
