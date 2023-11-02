# from flask import Flask, render_template, request

# import joblib
# from sklearn.feature_extraction.text import CountVectorizer
# import tweets,clean

# #__name__ == __main__
# app = Flask(__name__)

# #cv = CountVectorizer( max_features=10000, ngram_range=(1,2))
# cv = joblib.load('vectorizer1.pkl')
# classifier = joblib.load("model.pkl")

# dict = {0:'ENFJ',
#         1: 'ENFP',
#         2: 'ENTJ',
#         3: 'ENTP',
#         4: 'ESFJ',
#         5: 'ESFP',
#         6: 'ESTJ',
#         7: 'ESTP',
#         8: 'INFJ',
#         9: 'INFP',
#         10: 'INTJ',
#         11: 'INTP',
#         12: 'ISFJ',
#         13: 'ISFP',
#         14: 'ISTJ',
#         15: 'ISTP'}

# @app.route('/')
# def hello():
# 	return render_template("index.html")

# @app.route('/predict', methods=['POST'])
# def personality():
# 	if request.method == 'POST':
# 		twitter_handle = request.form['username']
# 		tweet = tweets.fetch(twitter_handle)
# 		tweet = clean.clean_text(tweet)
# 		res = classifier.predict(cv.transform([tweet]))
# 		##result = dict[str(res[0])]

# 	return render_template("index.html", personality = "Personality type is {}".format(res))

# if __name__ == '__main__':
# 	app.run(debug = True)