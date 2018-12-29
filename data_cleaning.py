import pandas as pd
import pandasql
import os

# The goal here is to take the 3 million tweets, and prune out all non-english tweets, and
# all retweets.

# Please Note a few datatype errors may show up as you run this, but they can be ignored since they are
# for columns we will not be using anyway. We only care about the tweet content and not where links
# are directed or anything like that



if not ('righttweets.csv' in os.listdir('./csv/') and 'lefttweets.csv' in os.listdir('./csv/')):
    first = True

    # Read in the file names for the tweets. Change tweet_dir to where you have the tweets stored
    tweet_dir = './tweets/'
    files = sorted(os.listdir('./tweets'))
    if 'README.md' in files:    
        files.remove('README.md')

    total_tweet_count = 0
    lw_tweet_count = 0
    rw_tweet_count = 0
    russian_tweet_count = 0

    for f in files:
        #print("Working on {}".format(f))
        raw_tweets = pd.read_csv(tweet_dir + f)
        total_tweet_count += len(raw_tweets)
        russian_tweet_count += len(raw_tweets.query('language == "Russian"'))
        english_tweets = raw_tweets.query('language == "English"')
        tweets_no_extra = pd.DataFrame()
        tweets_no_extra['content'] = english_tweets['content'].values
        # we have to use fillna since by default the post type column only is used to indicate retweets
        tweets_no_extra['post_type'] = english_tweets['post_type'].fillna('TWEET')
        tweets_no_extra['account_type'] = english_tweets['account_type'].values
        tweets_no_extra['account_category'] = english_tweets['account_category'].values
        tweets_no_extra['followers'] = english_tweets['followers'].values
        tweets_no_extra['following'] = english_tweets['following'].values

        # split the tweets into whether or not they were left or right wing
        english_tweets_no_rts_rwing = tweets_no_extra.query('account_category == "RightTroll"')
        english_tweets_no_rts_lwing = tweets_no_extra.query('account_category == "LeftTroll"')
        if first: # write them
            english_tweets_no_rts_rwing.to_csv('csv/righttweets.csv', index=False)
            english_tweets_no_rts_lwing.to_csv('csv/lefttweets.csv', index=False)
            first = False
        else:
            english_tweets_no_rts_rwing.to_csv('csv/righttweets.csv', index=False, mode='a', header=False)
            english_tweets_no_rts_lwing.to_csv('csv/lefttweets.csv', index=False, mode='a', header=False)
        lw_tweet_count += len(english_tweets_no_rts_lwing)
        rw_tweet_count += len(english_tweets_no_rts_rwing)
        
    print("Total Number of Tweets: {}".format(total_tweet_count))
    print("Number of Right Wing English tweets: {}".format(rw_tweet_count))
    print("Number of Left Wing English tweets: {}".format(lw_tweet_count))
    print("Number of Other Tweets: {}".format(total_tweet_count - lw_tweet_count - rw_tweet_count))
    print("Number of Russian Tweets: {}".format(russian_tweet_count))

else: 
    print("righttweets.csv and lefttweets.csv already exist! so skipping creating them")

