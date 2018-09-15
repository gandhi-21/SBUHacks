
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.cross_validation import train_test_split

stock = "AMD"
url = "https://api.iextrading.com/1.0/stock/AMD/chart/5y?filter=data,date,close"
urlInfo = 'https://api.iextrading.com/1.0/stock/"+stock+"/company'
closingprices = []
dates = []
data = pd.DataFrame(pd.read_json(url, orient="columns"))

#print(data)

for i in data:
    dates.append(data["date"])
    closingprices.append(data["close"])

#print(dates)
#print(closingprices)

# Latest Prices at the top
cp = list(reversed(closingprices))

#print(cp)

# Getting the trend scores

scores = []
with open("trends.txt") as f:
    data = f.readlines()
    scores = str(data).split(",")
    scores[0] = scores[0][2:]
    scores[-1] = scores[-1][:2]
    int_scores = []

# print(scores)

for score in scores:
    int_scores.append(int(score))

# print(int_scores)

# Split the data
x_train, x_test = train_test_split(closingprices, test_size=0.20)
y_train, y_test = train_test_split(int_scores, test_size=0.20)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
y_train = np.array(y_train)

xtdf = pd.DataFrame(data=x_train.T).loc[0:366]
xtedf = pd.DataFrame(data=x_test.T).loc[0:366]
ytdf = pd.DataFrame(y_train).loc[0:366]
ytedf = pd.DataFrame(y_test).loc[0:366]

# x_train.reshape(1260, 1)
# x_test.reshape(1260, 1)
# y_train.reshape(1461, 1)
# y_test.reshape(366, 1)

# np.transpose(x_train)
# np.transpose(x_test)

xtdf.truncate(before=0, after=200)
xtedf.truncate(before=0, after=200)
ytdf.truncate(before=0, after=200)
ytedf.truncate(before=0, after=200)

# print(xtdf)
# print(xtedf)
# print(ytdf)
# print(ytedf)

# print(x_train.shape, x_test.shape)

# np.delete(x_train, 0, 1)
# np.delete(y_train, 0, 1)

# print(x_train)

gbrt = ensemble.GradientBoostingRegressor(n_estimators=100).fit(xtdf, ytdf)
y_pred = gbrt.predict(x_test)
print(gbrt.score(xtdf, ytdf))
print(gbrt.score(xtedf, ytedf))

# # Trying to plot Deviance
#
# test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
#
# for i, y_pred in enumerate(gbrt.staged_predict(xtedf)):
#     test_score[i] = gbrt.loss_(ytedf, y_pred)
#
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Deviance')
# plt.plot(np.arange(params['n_estimators']))
