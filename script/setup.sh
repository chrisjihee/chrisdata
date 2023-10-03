# for running program
conda create -n WiseData-2023.08 python=3.10 -y
conda activate WiseData-2023.08
pip install -r requirements.txt
pip list

# for developing library
rm -rf chrisbase* chrislab*
git clone https://github.com/chrisjihee/chrisbase.git
git clone https://github.com/chrisjihee/chrislab.git
pip install --editable chrisbase
pip install --editable chrislab

#bash script/setup-mongodb.sh
#bash script/setup-elasticsearch8.sh
#bash script/setup-elasticsearch7.sh
