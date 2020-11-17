#!/bin/bash
apt-get install python3-pydot graphviz -y
/usr/bin/python3 -m pip install --upgrade pip
/usr/bin/python3 -m pip install scikit-learn pandas tqdm tables seaborn pydot_ng
