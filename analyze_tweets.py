import pandas as pd
import numpy as np
import pandasql
import os
import scipy.stats
import spacy



example_tweet = "Well, it's #telling #GeorgeSoros #legacy https://t.co/AHMTDEc78s"
example_tweet2 = "Well, it's telling #GeorgeSoros #legacy https://t.co/AHMTDEc78s"

# handily, all these tweets are formatted to any links at the end if there are any
def trim_tailing_link(tweet):
    result = tweet.split(" ")
    if "https://" in result[-1]:
        return (" ".join(result[0:-1]))
    else:
        return tweet

# Simple function that counts the number of hashtags in a tweet
def count_hashtags(tweet): 
    tweet_word_list = tweet.split(" ")
    hashtag_total = 0
    for word in tweet_word_list:
        if len(word) > 1 and word[0] == '#':
            hashtag_total += 1
    #result = [1 for i in tweet.split(" ") if i[0] == '#']

    return hashtag_total

rw_tweet_map = {}
lw_tweet_map = {}

def tally_hashtags(tweet, l_or_r):
    tweet_word_list = tweet.split(" ")

    if l_or_r == 'left':
        for word in tweet_word_list:
            if len(word) > 1 and word[0] == '#':
                if word.lower() in lw_tweet_map:
                    lw_tweet_map[word.lower()] += 1
                else:
                    lw_tweet_map[word.lower()] = 1
    elif l_or_r == 'right':
        for word in tweet_word_list:
            if len(word) > 1 and word[0] == '#':
                if word.lower() in rw_tweet_map:
                    rw_tweet_map[word.lower()] += 1
                else:
                    rw_tweet_map[word.lower()] = 1

rw_tweets_filename = './csv/righttweets.csv'
lw_tweets_filename = './csv/lefttweets.csv'

rw_tweets = pd.read_csv(rw_tweets_filename)
lw_tweets = pd.read_csv(lw_tweets_filename)

# trim off tailing links since they won't be useful for analysis in our context
rw_tweets['content'] = rw_tweets.apply(lambda row: trim_tailing_link(row['content']), axis=1)
lw_tweets['content'] = lw_tweets.apply(lambda row: trim_tailing_link(row['content']), axis=1)

# now that links are removed, we want to get rid of any tweets that were just links since
# the tweet text will now be empty.
#print("number of RW tweets before culling: {}".format(len(rw_tweets['content'])))
rw_tweets['length'] = rw_tweets.apply(lambda row: len(row['content']), axis=1)
rw_tweets = rw_tweets[rw_tweets['length'] > 5]
#print("number of RW tweets after culling: {}".format(len(rw_tweets['content'])))

#print("number of LW tweets before culling: {}".format(len(lw_tweets['content'])))
lw_tweets['length'] = lw_tweets.apply(lambda row: len(row['content']), axis=1)
lw_tweets = lw_tweets[lw_tweets['length'] > 5]
#print("number of LW tweets after culling: {}".format(len(lw_tweets['content'])))


rw_tweets['hashtag_count'] = rw_tweets.apply(lambda row: count_hashtags(row['content']), axis=1)
lw_tweets['hashtag_count'] = lw_tweets.apply(lambda row: count_hashtags(row['content']), axis=1)
#print(lw_tweets['hashtag_count'])
print("mean number of hashtags per tweet for right wing: {}".format(np.mean(rw_tweets['hashtag_count'])))
print("mean number of hashtags per tweet for left  wing: {}".format(np.mean(lw_tweets['hashtag_count'])))

print("variance in number of hashtags per tweet for right wing: {}".format(np.var(rw_tweets['hashtag_count'])))
print("variance in number of hashtags per tweet for left wing: {}".format(np.var(lw_tweets['hashtag_count'])))

rw_tweets.apply(lambda row: tally_hashtags(row['content'], 'right'), axis=1)
rw_tweet_sorted_list = sorted(rw_tweet_map, key=rw_tweet_map.get, reverse=True)

for i in range(25):
    print('{}: {}'.format(rw_tweet_sorted_list[i], rw_tweet_map[rw_tweet_sorted_list[i]]))

# Looking through the list of different hashtags, it's clear there are a lot of variants on MAGA
# so if we ignore whatever emojis that were accidentally tagged onto the end it will probably
# increase the count a lot
maga_count = 0
for i in rw_tweet_sorted_list:
    if "#maga" in i.lower():
        maga_count += rw_tweet_map[i]

print("Total corrected #MAGA count: {}".format(maga_count))

lw_tweets.apply(lambda row: tally_hashtags(row['content'], 'left'), axis=1)
lw_tweet_hashtag_sorted_list = sorted(lw_tweet_map, key=lw_tweet_map.get, reverse=True)

for i in range(100):
    print('{}: {}'.format(lw_tweet_hashtag_sorted_list[i], lw_tweet_map[lw_tweet_hashtag_sorted_list[i]]))

nlp = spacy.load('en_core_web_sm')
data = []

# RIP computer
for i in rw_tweets['content'].values[:25]:
    data.append(nlp(i))

data.append(nlp(trim_tailing_link(example_tweet)))
for i in data:
    for j in i.ents:
        print(j.text, j.label_)


