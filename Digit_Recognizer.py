import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

#Reading input data from csv file downloaded from kaggle
data=pd.read_csv("train.csv")

X=data.drop('label',axis=1)
y=data.loc[:,'label']


clf=MLPClassifier(solver='adam',alpha=0.001,hidden_layer_sizes=(256,80),random_state=42)
kf=KFold(n_splits=10,random_state=42)
scores = []
times = []
for train_ixs, test_ixs in kf.split(X):
    start_time = time.time()
    clf.fit(X.loc[train_ixs], y.loc[train_ixs])
    end_time = time.time()
    time_taken = end_time - start_time
    print("time taken: ",time_taken)
    times.append(time_taken)
    score = clf.score(X.loc[test_ixs], y.loc[test_ixs])
    print(score)
    scores.append(score)

print("mean score:", np.mean(scores))
