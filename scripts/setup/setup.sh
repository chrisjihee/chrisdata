# for running program
mamba create -n chrisdata python=3.10 -y
mamba activate chrisdata
pip install -r requirements.txt
pip list

# for developing library
rm -rf chrisbase* chrislab*
git clone git@github.com:chrisjihee/chrisbase.git
git clone git@github.com:chrisjihee/chrislab.git
pip install --editable chrisbase
pip install --editable chrislab

#bash script/setup-mongodb.sh
#bash script/setup-elasticsearch8.sh
#bash script/setup-elasticsearch7.sh
