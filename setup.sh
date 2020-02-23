#!/bin/bash
# run this script in the directory you want to work in. If you have already set up your virtual environment, then comment out the virtual environment section
# if you already have the data, then simply move all of the .csv files into a directory called tweets and the programs should work as expected.
virtualenv venv -p python3
source venv/bin/activate
pip install pandas
pip install pandasql
pip install matplotlib
pip install nltk
pip install textblob
python -m textblob.download_corpora
pip install scipy
pip install spacy
pip install jupyter
pip install seaborn
pip install emoji

mkdir tweets
mkdir csv

git clone https://github.com/fivethirtyeight/russian-troll-tweets.git
cp ./russian-troll-tweets/* ./tweets/
rm -rf ./russian-troll-tweets
rm ./tweets/README.md


