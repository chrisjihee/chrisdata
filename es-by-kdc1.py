import json
from pathlib import Path

from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from tqdm import tqdm

elastic_password = Path("cfg/elastic-pw.txt").read_text().strip().splitlines()[-1].strip()
elastic_index_name = "example_index"
elastic_host_info = "https://localhost:9200"
elastic_ca_certs = "elasticsearch/config/certs/http_ca.crt"
input_documents = "input/CHOSUN_2000.small.jsonl"
total_documents = 50


def _document_generator(documents_path):
    with open(documents_path) as fp:
        documents = list(map(lambda x: json.loads(x), fp.readlines()))
        for i, document in enumerate(documents):
            yield dict(
                index=i,
                id=document["id"],
                lang=document["lang"],
                date=document["date"],
                title=document["title"],
                body=document['body'],
                both=f"{document['title']} {document['body']}"
            )


def main():
    es: Elasticsearch = Elasticsearch(
        hosts=elastic_host_info,
        request_timeout=30,
        max_retries=10,
        retry_on_timeout=True,
        basic_auth=("elastic", elastic_password),
        verify_certs=True,
        ca_certs=elastic_ca_certs,
    )

    if es.indices.exists(index=elastic_index_name):
        es.indices.delete(index=elastic_index_name)
    print("Creating an index ...")
    es.indices.create(
        index=elastic_index_name,
        settings={
            "analysis": {
                "analyzer": {"my_analyzer": {"tokenizer": "my_tokenizer"}},
                "tokenizer": {"my_tokenizer": {"type": "ngram", "min_gram": "2", "max_gram": "2", }},
            },
            "index": {
                "number_of_shards": 1,
                "blocks": {"read_only_allow_delete": "false"},
            }
        },
        mappings={
            "properties": {
                "index": {"type": "integer"},
                "id": {"type": "keyword"},
                "lang": {"type": "keyword"},
                "date": {"type": "keyword"},
                "title": {"type": "keyword"},
                "body": {"type": "text", "analyzer": "my_analyzer"},
                "both": {"type": "text", "analyzer": "my_analyzer"},
            },
        },
        # ignore=400
    )

    print("Indexing documents ...")
    successes = 0
    pbar = tqdm(unit="docs", total=total_documents, desc="Indexing")
    for ok, action in streaming_bulk(client=es,
                                     actions=_document_generator(documents_path=input_documents),
                                     index=elastic_index_name,
                                     chunk_size=256, ):
        pbar.update(1)
        successes += ok
    pbar.close()

    print("Indexed %d/%d documents" % (successes, total_documents))
    es.indices.refresh(index=elastic_index_name)
    print(es.cat.indices(index=elastic_index_name, v=True).body.strip())

    query = "도쿄"
    nbest = 2
    response: ObjectApiResponse = es.search(
        index=elastic_index_name,
        query={
            "match": {
                "both": {"query": query}
            },
        },
        _source=("id", "title", "body"),
        size=nbest,
    )
    print(json.dumps(dict(response), ensure_ascii=False))
    response: ObjectApiResponse = es.search(
        index=elastic_index_name,
        query={
            "match": {
                "both": {"query": query}
            },
        },
        _source=("index", "id", "lang", "date", "title"),
        size=nbest,
    )
    print(json.dumps(dict(response), ensure_ascii=False))


if __name__ == "__main__":
    main()
