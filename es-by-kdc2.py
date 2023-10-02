from pathlib import Path

from elasticsearch import Elasticsearch

elastic_password = Path("cfg/eleastic-pw.txt").read_text().strip().splitlines()[-1].strip()
elastic_index_name = "example_index"
elastic_host_info = "https://localhost:9200"
elastic_ca_certs = "cfg/http_ca.crt"


def demo(es, index, query, beam_size=10, response_fields=("id", "body"), explain=False):
    response = es.search(
        index=index,
        explain=True if explain else False,
        query={
            "match": {"body": {"query": query}}
        },
        _source=response_fields,
        size=beam_size
    )
    print(response)


def main():
    es = Elasticsearch(
        hosts=elastic_host_info,
        request_timeout=30,
        max_retries=10,
        retry_on_timeout=True,
        basic_auth=("elastic", elastic_password),
        verify_certs=True,
        ca_certs=elastic_ca_certs,
    )
    demo(es, index=elastic_index_name, query="대통령", beam_size=10, explain=False)
    demo(es, index=elastic_index_name, query="대통령", beam_size=10, response_fields=("id",), explain=False)


if __name__ == "__main__":
    main()
