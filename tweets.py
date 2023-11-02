import tweepy as tw

consumer_key = "7NCzIYyE6v4rXHFgBfMjy6GqD"
consumer_secret = "mkD5V5WdViePkoiNdt5R4W0o8PJ8tojHXUxGgMzHhctp0rbRI6"
access_token = "1144102706333618176-wIUa115mxUw7n1IHmpFEW5RTSs81li"
access_token_secret = "ccwvQxyexrKQTpdzRJyxgF6nJratx2qvKzTbWrrMwgPYn"


def fetch(handle):
	# Authenticate
	auth = tw.OAuthHandler(consumer_key, consumer_secret)
	# Set Tokens
	auth.set_access_token(access_token, access_token_secret)
	# Instantiate API
	api = tw.API(auth, wait_on_rate_limit=True)

	res = api.user_timeline(screen_name=handle, count=100, include_rts=True)
	tweets = [tweet.text for tweet in res]
	text = ''.join(str(tweet) for tweet in tweets)
	return text
'''

import tweepy #https://github.com/tweepy/tweepy
import csv

#Twitter API credentials



def get_all_tweets(screen_name):
    #Twitter only allows access to a users most recent 3240 tweets with this method
    
    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    #initialize a list to hold all the tweepy Tweets
    alltweets = []  
    
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=1)
    
    #save most recent tweets
    alltweets.extend(new_tweets)
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    
    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print(f"getting tweets before {oldest}")
        
        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=1,max_id=oldest)
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        
        print(f"...{len(alltweets)} tweets downloaded so far")
    
    #transform the tweepy tweets into a 2D array that will populate the csv 
    #outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]
    
    #write the csv  
    with open(f'new_{screen_name}_tweets.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text"])
        writer.writerows(outtweets)
    
    pass

if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets("J_tsar")
'''