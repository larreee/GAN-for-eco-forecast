import tweepy
import csv
import json
import os
import pandas as pd
import numpy as np
import datetime as dt
import time
import sys
sys.path.append("../")
from SQL_functions import dataframe_to_sql, sql_to_dataframe


def authorize_Twitter(consumer_key, consumer_secret, access_key, access_secret):
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)
	return api

def choose_columns(df, columns):
	return df.reindex(columns, axis = 'columns')

def clean_tweets(tweet_df):
	for t in tweet_df.columns:
		try:
			tweet_df[t] = tweet_df[t].str.replace('\n','\\n ')
			tweet_df[t] = tweet_df[t].str.replace('\t','\\t ')
			tweet_df[t] = tweet_df[t].str.replace('\r','\\r ')
		except (KeyError, AttributeError):
			pass
	return tweet_df

def fix_time_format(df):
	for date_column in ['created_at', 'user.created_at', 'retweeted_status.created_at', 'retweeted_status.user.created_at', 'quoted_status.created_at', 'quoted_status.user.created_at', 'retweeted_status.quoted_status.created_at', 'retweeted_status.quoted_status.user.created_at']:
		try:
			df[date_column] = np.where(pd.isna(df[date_column]),df[date_column],df[date_column].astype(str))
			df[date_column] = df[date_column].str.replace(" \+\d{4}", '')
			df.loc[df[date_column].notna(), date_column] =\
				df.loc[df[date_column].notna(), date_column].apply( dt.datetime.strptime, args = ('%a %b %d %H:%M:%S %Y',)
					).apply( dt.datetime.strftime, args = ('%Y-%m-%d',))
		except KeyError:
			print('Missed date_column')		
	return df

def get_credentials():
	# load Twitter API credentials
	with open('twitter_credentials.json') as cred_data:
		info = json.load(cred_data)
		consumer_key = info['CONSUMER_KEY']
		consumer_secret = info['CONSUMER_SECRET']
		access_key = info['ACCESS_KEY']
		access_secret = info['ACCESS_SECRET']
	return (consumer_key, consumer_secret, access_key, access_secret)

def tweets_to_dataframe(tweets):
	json_data = [t._json for t in tweets]
	df = pd.io.json.json_normalize(json_data)

	df = clean_tweets(df)
	return df

def save_tweets_csv(tweets, file_name):
	# transforming the tweets into a 2D array that will be used to populate the csv
	# Does not use pandas, takes list
	outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode('utf-8')] for tweet in tweets]
	save_name = '../Data/'+ file_name + '_tweets.csv'
	# writing to the csv file

	attempts = 0
	while os.path.isfile(save_name) and attempts < 50:
		save_name  = '../Data/'+ file_name + '_tweets{}.csv'.format('_'+str(attempts))
		attempts += 1

	with open(save_name, 'w', encoding='utf8') as f:
		writer = csv.writer(f)
		writer.writerow(['id', 'created_at', 'text'])
		writer.writerows(outtweets)

def scrape_by_account(screen_name, max_iters = 1, tweets_in_iter = 200):
	consumer_key, consumer_secret, access_key, access_secret = get_credentials()
	api = authorize_Twitter(consumer_key, consumer_secret, access_key, access_secret)

	tweets = [] # List to hold tweets gathered
	new_tweets = api.user_timeline(screen_name= screen_name, count= tweets_in_iter) # Tweets have to be obtained 200 at a time
	tweets.extend(new_tweets) # saving the most recent tweets
	oldest_tweet = tweets[-1].id - 1 # save id of 1 less than the oldest tweet

	# grabbing tweets till none are left or till max_iters is reached
	iters = 0
	mod_iter = 1
	modulo = 6
	while (len(new_tweets) > 0) and (iters < max_iters):
		iters += 1
		if mod_iter%modulo == 0:
			print('____________ Gathered {} total number of tweets, now sleeping... ____________'.format(len(tweets)))
			time.sleep(305)
			mod_iter = 0
		try:
			new_tweets = api.user_timeline(screen_name= screen_name, count= tweets_in_iter, max_id= oldest_tweet, tweet_mode = 'extended') # The max_id param will be used subsequently to prevent duplicates
			tweets.extend(new_tweets) # save most recent tweets
			oldest_tweet = tweets[-1].id - 1 # id is updated to oldest tweet - 1 to keep track
			mod_iter += 1
		except tweepy.error.TweepError as e:
			print('Tweepy stopped \n')
			print(e)
			break
	print('______ no. Tweets={}, new tweets len= {}, iters= {} ____________'.format(len(tweets), len(new_tweets), iters))
	#save_tweets_csv(the_tweets, screen_name)
	frame = tweets_to_dataframe(tweets)
	return frame

def scrape_by_hashtag(hashtag, max_no = 15):
	consumer_key, consumer_secret, access_key, access_secret = get_credentials()
	api = authorize_Twitter(consumer_key, consumer_secret, access_key, access_secret)

	tweets = [] # List to hold tweets gathered

	try:
		for tweet in tweepy.Cursor(api.search, q='#' + hashtag, lang = 'en', rpp=100).items(max_no):
			tweets.append(tweet)
	except tweepy.error.TweepError as e:
		print('Tweepy stopped \n')
		print(e)
	if len(tweets)==0:
		print('No tweets')
		return False

	frame = tweets_to_dataframe(tweets)
	return frame

#### Main ####
from get_tweet_from_id import get_tweet_from_id

if __name__ == '__main__':
	reference_file = pd.ExcelFile('../references.xlsx')
	accs = pd.read_excel(reference_file, 'twitter_accounts')
	columns = pd.read_excel(reference_file, 'twitter_columns')

	# accs_todo = accs.loc[accs['API'].isna(), 'account_name']
	accs_todo = ['POTUS']

	print('____________ Refrence data loaded...____________')
	
	for username in accs_todo:
		try:
			# username = input("Please choose account to scrape: ")
			print('____________ Starting scrape for user: {}...____________'.format(username))


			#frame = scrape_by_hashtag('TRUMP', max_no = 15)
			frame = scrape_by_account(username, max_iters = 50)

			print('____________ Tweets scraped... ____________')

			frame = choose_columns(frame, columns['Name'])
			frame = fix_time_format(frame)

			print('____________ Tweets cleaned and formatted... ____________')

			frame.to_csv("check.csv")
			frame = dataframe_to_sql(frame, 'twitter_x_{}'.format(username))

			print('____________ Tweets saved to SQL ____________')
		except Exception as e:
			print(e, username, "API")


	print("----------------- Finished first API search, now starting ID to API ---------------------")
	# accs_todo = accs.loc[ (accs['WEB'] == 'DONE') & (accs['ID_to_API'].isna()), 'account_name']

	# for username in accs_todo:
	# 	try:
	# 		print('____________ Starting scrape for user: {} ____________'.format(username))

	# 		tweets = get_tweet_from_id(username)
	# 		frame = tweets_to_dataframe(tweets)

	# 		print('____________ Tweets scraped... ____________')


	# 		frame = choose_columns(frame, columns['Name'])
	# 		frame = fix_time_format(frame)

	# 		print('____________ Tweets cleaned and formatted... ____________')

	# 		frame = dataframe_to_sql(frame, 'twitter_{}'.format(username))

	# 		print('____________ Tweets saved to SQL ____________')
	# 	except Exception as e:
	# 		print(e, username, "API")