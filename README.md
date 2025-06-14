# chrisdata

Data processing tools for data analysis


## Installation

1. Install Miniforge
    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

2. Clone the repository
    ```bash
    rm -rf chrisdata*
    git clone https://github.com/chrisjihee/chrisdata.git
    cd chrisdata*
    ```

3. Create a new environment
    ```bash
    conda install -n base conda-forge::conda --all -y
    mamba create -n chrisdata python=3.12 -y
    mamba activate chrisdata
    ```

4. Install the required packages
    ```bash
    pip install -U -e .
    rm -rf chrisbase;    git clone https://github.com/chrisjihee/chrisbase.git;    pip install -e chrisbase
    rm -rf progiter;     git clone https://github.com/chrisjihee/progiter.git;     pip install -e progiter
    pip list | grep -E "mongo|search|Wiki|wiki|json|pydantic|chris|prog|Flask"
    ```

5. Install MongoDB
    ```bash
    mkdir -p mongodb; cd mongodb; mkdir -p data log
    if [ "$(uname)" = "Linux" ]; then
      aria2c https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2404-8.0.9.tgz
    elif [ "$(uname)" = "Darwin" ]; then
      aria2c https://fastdl.mongodb.org/osx/mongodb-macos-arm64-8.0.9.tgz
    fi
    tar zxvf mongodb-*.tgz --strip-components=1
    cd ..
    ```

6. Run MongoDB
    ```bash
    cd mongodb
    bin/mongod --config ../cfg/mongod-8800.yaml
    cd ..
    ```
    ```bash
    cd mongodb
    bin/mongod --config ../cfg/mongod-8801.yaml
    cd ..
    ```

7. Install Elasticsearch
    ```bash
    mkdir elasticsearch7; cd elasticsearch7
    if [ "$(uname)" = "Linux" ]; then
      aria2c https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.10-linux-x86_64.tar.gz
    elif [ "$(uname)" = "Darwin" ]; then
      aria2c https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.10-darwin-aarch64.tar.gz
    fi
    tar zxf elasticsearch-*.tar.gz --strip-components 1
    sed -i '' 's/#http.port: 9200/http.port: 9717/g' ./config/elasticsearch.yml
    echo "xpack.security.enabled: true" >> ./config/elasticsearch.yml
    cd ..
    ```

8. Link input data
    ```bash
    cd input
    ln -s /mnt/geo/data/wikidata .
    ln -s /mnt/geo/data/wikipedia .
    cd ..
    ```


## Execution

1. Show help
    ```bash
    python -m chrisdata.cli --help
    ```

    ```bash
    python -m chrisdata.cli wikipedia --help
    ```

    ```bash
    python -m chrisdata.cli wikidata --help
    ```

2. Run command
  * To crawl Wikipedia articles
    ```bash
    python -m chrisdata.cli wikipedia crawl --from-scratch --max-workers 3
    ```
    ```bash
    python -m chrisdata.cli wikipedia crawl --max-workers 12 --prog-interval 50 --input-name backup/kowiki-20250601-all-titles-in-ns0.txt
    ```
    ```bash
    python -m chrisdata.cli wikipedia crawl --max-workers 3 --prog-interval 10 --input-name kowiki-20250601-all-titles-in-ns0-failed.tsv
    ```

  * To parse Wikipedia articles
    ```bash
    python -m chrisdata.cli wikipedia parse --input-file-home input/Wikipedia --input-file-name kowiki-20250601-all-titles-in-ns0.jsonl --input-total 1556201 --output-file-name kowiki-20250601-parse.jsonl --output-table-reset
    ```

  * To convert Wikipedia articles
    ```bash
    python -m chrisdata.cli wikipedia convert
    ```

  * To parse Wikidata dump
    ```bash
    python -m chrisdata.cli wikidata parse
    ```

  * To filter Wikidata entities
    ```bash
    python -m chrisdata.cli wikidata filter
    ```

  * To convert Wikidata entities
    ```bash
    python -m chrisdata.cli wikidata convert
    ```


## Reference

* https://pypi.org/project/chrisdata/
* https://github.com/chrisjihee/chrisdata/
* https://github.com/chrisjihee/chrisbase/
