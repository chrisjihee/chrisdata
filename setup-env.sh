#!/bin/bash
# conda
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# basic
mamba create -n WiseData python=3.11 -y; mamba activate WiseData
pip install -U -r requirements.txt

# chrisbase
rm -rf chrisbase*; git clone git@github.com:chrisjihee/chrisbase.git
pip install -U -e chrisbase*

# list
pip list | grep -E "search|Wiki|wiki|json|chris"
