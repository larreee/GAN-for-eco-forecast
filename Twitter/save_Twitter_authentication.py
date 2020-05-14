import json

# A dictionary for credentials

twitter_cred = dict()

twitter_cred['CONSUMER_KEY'] = 'zRDHziw7p3iNna9r2DhDrTodG'
twitter_cred['CONSUMER_SECRET'] = '4YTEm7ykW8mjx8yTsyGfQthulReMgrPJ7quuiuek8Lh3dlsNFU'
twitter_cred['ACCESS_KEY'] = '2654564066-tX4xKCN1szvGJpeDVxDya8TYi43yEW8TB1Kq5uk'
twitter_cred['ACCESS_SECRET'] = 'VIB5WXzojHyWvytxViNwW6a6RKT0G2onWoCbYHrr7dCwl'

# Save the information to a json so that it can be reused in code without exposing
# the secret info to public

with open('twitter_credentials.json', 'w') as secret_info:
	json.dump(twitter_cred, secret_info, indent=4, sort_keys=True)