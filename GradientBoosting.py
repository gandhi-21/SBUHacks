
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# import urllib
#
# import pytrends
# import pytrends.request as TrendReq
# from sklearn.utils import shuffle
stock = "AMD"
url = "https://api.iextrading.com/1.0/stock/AMD/chart/5y?filter=data,date,close"
urlInfo = 'https://api.iextrading.com/1.0/stock/"+stock+"/company'
closingprices = []
dates = []
data = pd.DataFrame(pd.read_json(url, orient="columns"))
for i in data:
    dates.append(data["date"])
    closingprices.append(data["close"])

cp = list(reversed(closingprices))

scores = []
with open("trends.txt") as f:
    data = f.readlines()
    scores = str(data).split(",")
    scores[0] = scores[0][2:]
    scores[-1] = scores[-1][:2]
    int_scores = []

    for score in scores:
        int_scores.append(int(score))
# pytrends = TrendReq(hl='en-US', tz=360)
# kw_list = ["AMD"]
# pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', )

# scores = []
# # with open("trends.txt") as f:
# #     data = f.readlines()
# #     scores = str(data).split(",")
# ratings = scores

#############################################################################################
## Using Pandas fro the data

DATES = np.array(dates)
CLOSINGPRICES = np.array(cp)
TRENDSCORES = np.array(int_scores)
DatesDF = pd.DataFrame(data=DATES.T)
pricesDF = pd.DataFrame(data=CLOSINGPRICES.T)
trendsDF = pd.DataFrame(data=TRENDSCORES)

completeDF = pd.DataFrame()
completeDF = completeDF.assign(Date=DATES[0])
completeDF = completeDF.assign(ClosingPrice=CLOSINGPRICES[0])
completeDF = completeDF.assign(GoogleTrendsScore=trendsDF[0])
# print(completeDF)

## Gradient Boosting Algorithm from Scikit-Learn
#############################################################################################

tempDF = completeDF
# est = GradientBoostingRegressor()

train_x = [tempDF["ClosingPrice"][:-252]]
test_x = [tempDF["ClosingPrice"][-252:]]
train_y = [tempDF["GoogleTrendsScore"][:-252]]
test_y = [tempDF["GoogleTrendsScore"][-252:]]

plt.scatter(train_x, train_y, color="black")
plt.ylabel("StockPrice")
plt.xlabel("Google Trend Score")
plt.show()

params = {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 4, 'random_state': 0, 'loss': 'ls'}
gbt = GradientBoostingRegressor(params).fit(train_x, train_y)
mse = mean_squared_error(test_y, GradientBoostingRegressor.staged_predict(test_x))
print(mse)

test_score = np.zeros((params['n_estimators']), dtype=np.float64)

for i, y_pred in enumerate(GradientBoostingRegressor.staged_predict(test_x)):
    test_score[i] = gbt.loss_(test_y, y_pred)
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, gbt.train_score_, 'b-', label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
plt.show()

# X, y = 
# X_train, X_test = X[:200], X[200:]
# y_train, y_test = y[:200], y[200:]

# params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 1, 'random_state': 0, 'loss': 'ls'}
# est = GradientBoostingRegressor(**params).fit(X_train, y_train)
# mse = mean_squared_error(y_test, est.predict(X_test))
# print(mse)

# test_score = np.zeros((params['n_estimators']), dtype=np.float64)

# for i, y_pred in enumerate(est.staged_predict(X_test)):
#     test_score[i] = est.loss_(y_test, y_pred)
# plt.figure(figsize=(12,6))
# plt.subplot(1, 2, 1)
# plt.title('Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, est.train_score_, 'b-', label='Training Set Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Set Deviance')
# plt.legend(loc='upper right')
# plt.xlabel('Boosting Iterations')
# plt.ylabel('Deviance')
# plt.show()
# # feature_importance = clf.feature_importances_
# # feature_importance 100.0 * (feature_importance / feature_importance.max())
# # sorted_idx = np.argsort(feature_importance)
# # pos = np.arange(sorted_idx.shape[0]) + .5
# # plt.subplot(1, 2, 2)
# # plt.barh(pos, feature_importance[sorted_idx], align='center')
# # plt.yticks(pos, ) 