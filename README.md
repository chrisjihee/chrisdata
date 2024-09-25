# WiseData

Data processing tools for data analysis.

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
    pip install -U -r requirements.txt
    ```

4. Install editing packages
    ```bash
    rm -rf chrisbase*; git clone git@github.com:chrisjihee/chrisbase.git
    rm -rf chrislab*; git clone git@github.com:chrisjihee/chrislab.git
    pip install -U -e chrisbase*
    pip install -U -e chrislab*
    ```

5. List installed packages
    ```bash
    pip list | grep -E "search|Wiki|wiki|json|chris|Data"
    ```

## Reference

- https://github.com/chrisjihee/WiseData-2023.08/
- https://github.com/chrisjihee/chrisbase/
