#!/bin/bash

virtualenv venv -p python3
source venv/bin/activate
pip install pandas
pip install nltk
pip install textblob
python -m textblob.download_corpora
pip install scipy
pip install spacy
pip install jupyter
pip install emoji

mkdir tweets
mkdir csv

git clone https://github.com/fivethirtyeight/russian-troll-tweets.git
cp ./russian-troll-tweets/* ./tweets/
rm -rf ./russian-troll-tweets
rm ./tweets/README.md


