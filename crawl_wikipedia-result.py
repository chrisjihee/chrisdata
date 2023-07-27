import logging

from chrisbase.io import configure_unit_logger, LoggingFormat
from chrisbase.util import MongoDB
from crawl_wikipedia import ProcessResult

logger = logging.getLogger(__name__)
configure_unit_logger(logging.INFO, fmt=LoggingFormat.CHECK_24)

indent = None
with MongoDB(db_name="WiseData", tab_name="crawl_wikipedia", clear_table=False) as mongo:
    logger.info(f"#result: {mongo.table.count_documents({})}")
    for row in mongo.table.find({}).sort("_id"):
        res = ProcessResult.from_dict(row)
        logger.info(res.to_json(ensure_ascii=False, indent=indent))
    logger.info(f"#result: {mongo.table.count_documents({})}")
