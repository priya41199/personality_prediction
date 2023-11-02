import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
df = pd.read_csv("mbti_1.csv")
df.head()

sw = set(stopwords.words('english'))
ps = PorterStemmer()
def clean_text(sample):
  sample = sample.lower()

  sample = re.sub("[^a-zA-Z]+", " ", sample)

  sample = sample.split(" ")
  sample = [re.sub("http.*","",s) for s in sample if s not in sw]
  sample = [ps.stem(s) for s in sample]
  sample = " ".join(sample)
  return sample
df['cleaned_posts'] = df['posts'].apply(clean_text)
x = df['cleaned_posts']


cv = CountVectorizer( max_features=10000, ngram_range=(1,2))
cv.fit(x)
x=cv.transform(x)

tfidf = TfidfTransformer()
x = tfidf.fit_transform(x)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = df['type'].values
y = le.fit_transform(y)

y[:100]

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,shuffle=True,test_size=0.3)
print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)
type(xtrain)
import numpy as np

np.unique(ytrain)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

cv = CountVectorizer( max_features=10000, ngram_range=(1,2))
cv.fit(xtrain)
x_train=cv.transform(xtrain)

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

classifier=XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc_ovr_weighted',n_jobs=-1,cv=5,verbose=3)

from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(x_train,ytrain)
timer(start_time)

random_search.best_estimator_

random_search.best_params_

classifier=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.1,
              learning_rate=0.1, max_delta_step=0, max_depth=4,
              min_child_weight=7, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

classifier.fit(x_train,ytrain)

from sklearn.model_selection import cross_val_score

score=cross_val_score(classifier,x_train,ytrain,cv=10)

score

score.mean()
pred = classifier.predict(cv.transform(xtest))

from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(ytest,pred)
cm

import seaborn as sn
import matplotlib.pyplot as plt

plt.figure(figsize = (10,10))
sn.heatmap(cm, annot=True)

print('classification_report :\n',classification_report(ytest, pred))

