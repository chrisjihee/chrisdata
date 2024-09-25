# WiseData

Data processing tools for data analysis

## Installation

1. Install Miniforge
   ```bash
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```

2. Create a new environment
   ```bash
   mamba create -n WiseData python=3.11 -y; mamba activate WiseData
   ```

3. Clone the repository and install the package
   ```bash
   rm -rf WiseData*
   git clone git@github.com:chrisjihee/WiseData-2023.08.git
   pip install -U -e WiseData*
   ```

4. Install editing packages
   ```bash
   cd WiseData*
   rm -rf chrisbase*; git clone git@github.com:chrisjihee/chrisbase.git
   rm -rf chrislab*; git clone git@github.com:chrisjihee/chrislab.git
   pip install -U -e chrisbase*
   pip install -U -e chrislab*
   ```

5. List installed packages
   ```bash
   pip list | grep -E "search|Wiki|wiki|json|chris|Data"
   ```

6. Install MongoDB
   ```bash
   bash script/setup-mongodb.sh
   ```

7. Install Elasticsearch
   ```bash
   bash script/setup-elasticsearch7.sh
   ```

## Execution

1. Show help
   ```bash
   chrisdata --help
   ```

2. Run command
   * To check local IP addresses
      ```bash
      chrisdata hello-chrisdata
      ```
   * To crawl Wikipedia articles
      ```bash
      chrisdata hello-chrisdata
      ```
   * To parse Wikipedia articles
      ```bash
      chrisdata hello-chrisdata
      ```
   * To parse Wikidata dump
      ```bash
      chrisdata hello-chrisdata
      ```

## Reference

* https://github.com/chrisjihee/WiseData-2023.08/
* https://github.com/chrisjihee/chrisbase/
