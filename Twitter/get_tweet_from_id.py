import tweepy
import json
from tweepy import TweepError
from time import sleep
import math
import pandas as pd
import sys
from psycopg2.errors import InvalidTextRepresentation, BadCopyFileFormat
sys.path.append("../")
from SQL_functions import dataframe_to_sql, delete_duplicates
from get_ids_from_sql import get_ids_from_sql

def pre_proc(tweet_json):
    processed_tweets = []
    for tweet in tweet_json:
        processed_tweets.append(tweet.strip('"'))
    return processed_tweets

def get_tweet_from_id(user):
    from scrape_Twitter import authorize_Twitter, get_credentials

    consumer_key, consumer_secret, access_key, access_secret = get_credentials()
    api = authorize_Twitter(consumer_key, consumer_secret, access_key, access_secret)

    user = user.lower()

    with open('{}_all_ids.json'.format(user)) as f:
        json_ids = set(json.load(f))

    sql_ids = set(get_ids_from_sql('x_{}'.format(user)))
    ids = list(json_ids.difference(sql_ids))

    print('total ids: {}'.format(len(ids)))

    all_data = []
    start = 0
    end = 100
    limit = len(ids)
    i = math.ceil(limit / 100)

    tweets = []
    for go in range(i):
        print('currently getting {} - {}'.format(start, end))
        sleep(6)  # needed to prevent hitting API rate limit
        id_batch = ids[start:end]
        start += 100
        end += 100
        try:
            for tweet in api.statuses_lookup(id_batch, tweet_mode = 'extended'):
                tweets.append(tweet)
        except TweepError as e:
            print('Tweepy stopped \n')
            print(e)
            break
    return tweets

if __name__ == '__main__':
    from scrape_Twitter import tweets_to_dataframe, choose_columns, fix_time_format
    pd.set_option('display.max_colwidth',-1)
    reference_file = pd.ExcelFile('../references.xlsx')
    accs = pd.read_excel(reference_file, 'twitter_accounts')
    searches = pd.read_excel(reference_file, 'twitter_searches')
    columns = pd.read_excel(reference_file, 'twitter_columns')

    print('____________ Refrence data loaded...____________')

    #accs_todo = accs.loc[ (accs['WEB'] == 'DONE') & (accs['ID_to_API'].isna()), 'account_name']
    accs_todo = accs.loc[accs['Extra_run'].notna(), 'account_name']

    for a in accs_todo:
        print('____________ Starting scrape for: {} ____________'.format(a))

        tweets = get_tweet_from_id(a)
        frame = tweets_to_dataframe(tweets)

        print('____________ Tweets scraped... ____________')

        frame = choose_columns(frame, columns['Name'])
        frame = fix_time_format(frame)

        print('____________ Tweets cleaned and formatted... ____________')
        
        try:
            frame = dataframe_to_sql(frame, 'twitter_x_{}'.format(a))
            print('____________ Tweets saved to SQL ____________')

            delete_duplicates('twitter_x_{}'.format(a.lower()))
        
        except (InvalidTextRepresentation, BadCopyFileFormat):
            frame.to_csv("saved_{}.csv".format(a))
        print('____________ Cleaned for duplicates ____________')
