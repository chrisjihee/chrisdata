mamba create --name chrisdata python=3.11 -y; mamba activate chrisdata;
rm -rf build dist src/*.egg-info;
pip install build twine;
python3 -m build;
python3 -m twine upload dist/*;
rm -rf build dist src/*.egg-info;

sleep 3; clear;
mamba create --name chrisdata python=3.11 -y; mamba activate chrisdata;
sleep 5; clear;
pip install --upgrade chrisdata; pip list;
