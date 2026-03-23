uv venv env --python 3.12 --python-preference only-managed --clear
source env/bin/activate
rm -rf build dist src/*.egg-info
uv pip install build twine;
python3 -m build
python3 -m twine upload dist/*
rm -rf build dist src/*.egg-info

sleep 3; clear
uv venv env --python 3.12 --python-preference only-managed --clear
source env/bin/activate
sleep 5; clear;
uv pip install --upgrade chrisdata
uv pip list
