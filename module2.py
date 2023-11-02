import numpy as np
import pandas as pd
import os
'''for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
##%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from xgboost.sklearn import XGBClassifier 
#from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
#from sklearn.linear_model import SGDClassifier
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC

train = pd.read_csv('mbti_1.csv', header=0)
train.head(10)

train_feature = train['posts'].values.astype('U')
train_target = train['type']

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english').fit(train_feature)
train_feature = vect.transform(train_feature)
#print("train_feature with min_df:\n{}".format(repr(train_feature)))

X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0, shuffle=True)

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train, y_train)
print('='*20)
print('RandomForestClassifier')
print(f'accuracy of train set: {rf.score(X_train, y_train)}')
print(f'accuracy of test set: {rf.score(X_test, y_test)}')

#GRADIENT BOOSTING
gbc = GradientBoostingClassifier(n_estimators=200)
gbc.fit(X_train, y_train)
print('Gradient Boosting Classifier')
print(f'accuracy of train set: {gbc.score(X_train, y_train)}')
print(f'accuracy of test set: {gbc.score(X_test, y_test)}')

#ENSEMBLING
estimators=[]
estimators.append(('randomforest',rf))
estimators.append(('gradientboosting',gbc))
vc=VotingClassifier(estimators)
vc.fit(X_train,y_train)
print('Ensembled Classifier')
print(f'accuracy of train set: {vc.score(X_train, y_train)}')
print(f'accuracy of test set: {vc.score(X_test, y_test)}')

import joblib

joblib.dump(vc,'model.pkl')
joblib.dump(vect,'vectorizer1.pkl')