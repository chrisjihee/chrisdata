import json

from elasticsearch.helpers import streaming_bulk
from tqdm import tqdm
from pathlib import Path

from elasticsearch import Elasticsearch

elastic_password = Path("cfg/eleastic-pw.txt").read_text().strip().splitlines()[-1].strip()
elastic_index_name = "example_index"
elastic_host_info = "https://localhost:9200"
elastic_ca_certs = "cfg/http_ca.crt"


def _document_generator(documents_path, title_concat=False):
    with open(documents_path) as fp:
        documents = list(map(lambda x: json.loads(x), fp.readlines()))

        for i, document in enumerate(documents):
            yield dict(
                id=document["id"],
                index=i,
                title=document["title"],
                body=f"{document['title']} {document['body']}" if title_concat else document['body']
            )


def run_document_indexing(es, index_name, documents_path, num_documents, title_concat=False):
    print("Creating an index ...")
    es.indices.create(
        index=index_name,
        settings={
            "analysis": {
                "analyzer": {"my_analyzer": {"tokenizer": "my_tokenizer"}},
                "tokenizer": {"my_tokenizer": {"type": "ngram", "min_gram": "2", "max_gram": "2", }}
            },
            "index": {
                "number_of_shards": 1,
                "blocks": {"read_only_allow_delete": "false"},
            }
        },
        mappings={
            "properties": {
                "id": {"type": "keyword"},
                "index": {"type": "integer"},
                "title": {"type": "keyword"},
                "body": {"type": "text", "analyzer": "my_analyzer"}
            },
        },
        # ignore=400
    )

    print("Indexing documents ...")
    successes = 0
    pbar = tqdm(unit="docs", total=num_documents, desc="Indexing")
    for ok, action in streaming_bulk(client=es,
                                     actions=_document_generator(documents_path=documents_path,
                                                                 title_concat=title_concat),
                                     index=index_name,
                                     chunk_size=256, ):
        pbar.update(1)
        successes += ok
    pbar.close()

    print("Indexed %d/%d documents" % (successes, num_documents))

    es.indices.refresh(index=index_name)
    print(es.cat.indices(index=index_name, v=True))


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
    run_document_indexing(es, index_name=elastic_index_name,
                          documents_path="input/CHOSUN_2000.small.jsonl",
                          num_documents=50,
                          title_concat=True)


if __name__ == "__main__":
    main()
