python -m chrisdata.cli wikidata filter --output-table-reset
python -m chrisdata.cli wikidata parse --input-limit 30000000 --output-table-reset
python -m chrisdata.cli wikidata parse --input-start 30000000 --input-limit 30000000
python -m chrisdata.cli wikidata parse --input-start 60000000 --input-limit 30000000
python -m chrisdata.cli wikidata parse --input-start 90000000
