import os
import pandas as pd
import numpy as np
from multiprocessing import Process
from time import sleep
from time import time

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from scipy.stats import ttest_ind, shapiro

import string
import re
import random

csv_dir = './csv/'

csv_files = os.listdir(csv_dir)

# get rid of leading RT's, hashtags, punctuation, and @'s
def remove_cruft(tweet):
    words = tweet.split(" ")

    if (words[0]) == "RT":
        words = words[1:]


    if "https://" in words[-1] or "http://" in words[-1]:
        words = words[0:-1]
    
    #remove hashtags, handles, and punctuation
    words = [w for w in words if not ('#' in w) or ('@' in w)]
    words = [w.strip(string.punctuation) for w in words]
    words = list(filter(None, words)) # get rid of empty strings that can result from previous removals

    clean_tweet = " ".join(words)

    # remove emojis. the pattern is from stack exchange because unicode is a hellish nightmare
    reg = re.compile(u'['
    u'\U0001F300-\U0001F64F'
    u'\U0001F680-\U0001F6FF'
    u'\u2600-\u26FF\u2700-\u27BF]+', 
    re.UNICODE)
    clean_tweet = reg.sub(r'', clean_tweet).strip()

    return clean_tweet

def analyze_sentiment(tweet):
    blob = TextBlob(tweet)
    return blob.sentiment.polarity


good_to_go = True

if not('lefttweets.csv' in csv_files and 'righttweets.csv' in csv_files):
    print('You need to run data_cleaning.py before you can do anything with this file!!!!!')
    good_to_go = False

if good_to_go:
    if not('cleaned_lefttweets.csv' in csv_files and 'cleaned_righttweets.csv' in csv_files):
        print('First time running sentiment_analysis.py. Some prep needs to be done before we can start!')
        lw_tweets = pd.read_csv(csv_dir + "lefttweets.csv")
        rw_tweets = pd.read_csv(csv_dir + "righttweets.csv")

        # Similar to in analyze_tweets.py, we will clean the tweets, then remove anything shorter
        # than 5 characters since analyzing sentiment on that short of a sentence is pointless.
        rw_tweets['content'] = rw_tweets.apply(lambda row: remove_cruft(row['content']), axis=1)
        lw_tweets['content'] = lw_tweets.apply(lambda row: remove_cruft(row['content']), axis=1)

        # now that links are removed, we want to get rid of any tweets that were just links since
        # the tweet text will now be empty.
        print("number of RW tweets before culling: {}".format(len(rw_tweets['content'])))
        rw_tweets['length'] = rw_tweets.apply(lambda row: len(row['content']), axis=1)
        rw_tweets = rw_tweets[rw_tweets['length'] > 5]
        print("number of RW tweets after culling: {}".format(len(rw_tweets['content'])))

        print("number of LW tweets before culling: {}".format(len(lw_tweets['content'])))
        lw_tweets['length'] = lw_tweets.apply(lambda row: len(row['content']), axis=1)
        lw_tweets = lw_tweets[lw_tweets['length'] > 5]
        print("number of LW tweets after culling: {}".format(len(lw_tweets['content'])))

        lw_tweets.to_csv(csv_dir + "cleaned_lefttweets.csv", index=False)
        rw_tweets.to_csv(csv_dir + "cleaned_righttweets.csv", index=False)



    if not('analyzed_lefttweets.csv' in csv_files and 'analyzed_righttweets.csv' in csv_files):
        
        print("Running sentiment analysis on the tweets. Go and get some coffee or something because this is going to take a while")

        # running the two analyses on two different processors so it doesn't 5 minutes
        def analyze_lw_thread():
            print("Starting analysis of LW Tweets")
            lw_tweets_clean = pd.read_csv(csv_dir + "cleaned_lefttweets.csv")
            lw_tweets_clean['sent_polarity'] = lw_tweets_clean.apply(lambda row: analyze_sentiment(row['content']), axis=1)
            lw_tweets_clean.to_csv(csv_dir + "analyzed_lefttweets.csv")
            print("Done analyzing LW tweets")

        def analyze_rw_thread():
            print("Starting analysis of RW Tweets")
            rw_tweets_clean = pd.read_csv(csv_dir + "cleaned_righttweets.csv")
            rw_tweets_clean['sent_polarity'] = rw_tweets_clean.apply(lambda row: analyze_sentiment(row['content']), axis=1)
            rw_tweets_clean.to_csv(csv_dir + "analyzed_righttweets.csv")
            print("Done analyzing RW tweets")
        
        t = time()
        p1 = Process(target=analyze_lw_thread)
        p2 = Process(target=analyze_rw_thread)
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        while p1.is_alive() or p2.is_alive():
            sleep(1)
        print("process completed in {} seconds".format(time() - t))


    lw_tweets = pd.read_csv(csv_dir + "analyzed_lefttweets.csv")
    rw_tweets = pd.read_csv(csv_dir + "analyzed_righttweets.csv")

    print('Polarity for left wing tweets: {}'.format(np.mean(lw_tweets['sent_polarity'])))
    print('Polarity for right wing tweets: {}'.format(np.mean(rw_tweets['sent_polarity'])))

    lw_polarity = lw_tweets['sent_polarity'].values
    rw_polarity = rw_tweets['sent_polarity'].values

    lw_population_sample = random.sample(range(len(lw_tweets)), 10000)
    rw_population_sample = random.sample(range(len(rw_tweets)), 10000)

    lw_sample_polarity = [lw_polarity[i] for i in lw_population_sample]
    rw_sample_polarity = [rw_polarity[i] for i in rw_population_sample]

    #print("\nTest for normality among the two random samples of LW and RW tweets")
    #print(shapiro(lw_sample_polarity))
    #print(shapiro(rw_sample_polarity))

    print('\nTest to see if polarity in sample of LW tweets is significantly different than RW tweets')
    t, p = ttest_ind(lw_sample_polarity, rw_sample_polarity, equal_var=False)
    print('t-value: {}\np-value: {}'.format(t, p))
    if p < 0.05:
        print("Reject Null Hypothesis; the LW tweets have significantly different polarity than the right wing tweets")
    else: print ("Do not reject Null Hypothesis: Polarity is not significantly different among tweets")

    hillary_tweets = []
    hillary_polarity = []

    def find_hillary (tweet, polarity):
        if "hillary" in tweet.lower() or "clinton" in tweet.lower():
            hillary_tweets.append(tweet)
            hillary_polarity.append(polarity)
            return True
        return False
    rw_copy = rw_tweets
    rw_to_remove = rw_copy.apply(lambda row: find_hillary(row['content'], row['sent_polarity']), axis=1)
    to_remove = []
    for i in range(len(rw_copy)):
        if rw_to_remove[i]:
            to_remove.append(i)
    rw_copy.drop(to_remove, inplace=True)
    #print(rw_copy)

    print('\nNumber of Hillary Clinton-related tweets: {}'.format(len(hillary_tweets)))
    print('Average Polarity for Hillary Clinton-related tweets: {}'.format(np.mean(hillary_polarity)))

    donald_tweets = []
    donald_polarity = []

    def find_donald (tweet, polarity):
        if "donald" in tweet.lower() or "trump" in tweet.lower():
            donald_tweets.append(tweet)
            donald_polarity.append(polarity)

    rw_tweets.apply(lambda row: find_donald(row['content'], row['sent_polarity']), axis=1)

    print('\nNumber of Donald Trump-related tweets: {}'.format(len(donald_tweets)))
    print('Average Polarity for Donald Trump-related tweets: {}'.format(np.mean(donald_polarity)))

    
    h_random_tweet_selection = random.sample(range(len(hillary_polarity)), 10000)
    d_random_tweet_selection = random.sample(range(len(donald_polarity)), 10000)

    hillary_sample_polarity = [hillary_polarity[i] for i in h_random_tweet_selection]
    donald_sample_polarity = [donald_polarity[i] for i in d_random_tweet_selection]

    #print("Test for normality among the two random samples of Clinton and Trump-mentioning tweets")
    #print(shapiro(lw_random_tweet_selection))
    #print(shapiro(rw_random_tweet_selection))

    t, p = ttest_ind(hillary_sample_polarity, donald_sample_polarity, equal_var=False)
    print('\nTest to see if tweets mentioning Hillary Clinton are significantly different in terms of polarity than Donald Trump')
    print('t-value: {}\np-value: {}'.format(t, p))

    t, p = ttest_ind(hillary_sample_polarity, rw_sample_polarity, equal_var=False)
    print('\nTest to see if tweets mentioning Hillary Clinton are significantly different in terms of polarity than random RW tweets')
    print('t-value: {}\np-value: {}'.format(t, p))

    print('\nTests on Left Wing Tweets')
    hillary_tweets_lw = []
    hillary_polarity_lw = []

    def find_hillary2 (tweet, polarity):
        if "hillary" in tweet.lower() or "clinton" in tweet.lower():
            hillary_tweets_lw.append(tweet)
            hillary_polarity_lw.append(polarity)

    lw_tweets.apply(lambda row: find_hillary2(row['content'], row['sent_polarity']), axis=1)
    print('\nNumber of LW Hillary Clinton-related tweets: {}'.format(len(hillary_tweets_lw)))
    print('Average Polarity for LW Hillary Clinton-related tweets: {}'.format(np.mean(hillary_polarity_lw)))

    h_random_tweet_selection = random.sample(range(len(hillary_polarity_lw)), len(hillary_polarity_lw))
    lw_random_tweet_selection = random.sample(range(len(lw_polarity)), len(hillary_polarity_lw))

    h_sample_polarity = [hillary_polarity[i] for i in h_random_tweet_selection]
    lw_sample_polarity = [lw_polarity[i] for i in lw_random_tweet_selection]
    print(shapiro(h_sample_polarity))
    print(shapiro(lw_sample_polarity))

    t, p = ttest_ind(h_sample_polarity, lw_sample_polarity, equal_var=False)

    print('\nTest to see if tweets mentioning Hillary Clinton are significantly different in terms of polarity than random LW tweets')
    print('t-value: {}\np-value: {}'.format(t, p))
