
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

xtdf = pd.DataFrame(data=x_train.T).loc[0:200]
xtedf = pd.DataFrame(data=x_test.T).loc[0:200]
ytdf = pd.DataFrame(y_train).loc[0:200]
ytedf = pd.DataFrame(y_test).loc[0:200]

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


import codecs
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# GRABS DATA FROM STOCK API AND PUTS IT IN DATAFRAME
stock = "AMZN"
url = "https://api.iextrading.com/1.0/stock/"+stock+"/chart/5y?filter=date,close"
urlInfo = "https://api.iextrading.com/1.0/stock/"+stock+"/company"
closingprices = []
dates = []
data = pd.DataFrame(pd.read_json(url, orient="columns"))
for i in data:
    dates.append(data["date"])
    closingprices.append(data["close"])

# REVERSES SO THE LATEST IS ON TOP OF LIST
cp = list(reversed(closingprices))

# READS FROM TRENDS.TXT TO OBTAIN GOOGLE TRENDS SCORES (FROM NODEJS FILE)
scores = []
with open("Trends.txt") as f:
    data = f.readlines()
    scores = str(data).split(",")
    scores[0] = scores[0][2:]
    scores[-1] = scores[-1][:2]
    int_scores = []

    for score in scores:
        int_scores.append(int(score))



# CONVERTS LISTS FOR EACH DATASET TO NUMPY ARRAY
DATES = np.array(dates)
CLOSINGPRICES = np.array(cp)
TRENDSCORES = np.array(int_scores)
DatesDF = pd.DataFrame(data=DATES.T)
pricesDF = pd.DataFrame(data=CLOSINGPRICES.T)
trendsDF = pd.DataFrame(data=TRENDSCORES.T)

# PUTS ALL THE DATAFRAME COLUMNS TOGETHER
completeDF = pd.DataFrame()
completeDF = completeDF.assign(Date=DATES[0])
completeDF = completeDF.assign(ClosingPrice=CLOSINGPRICES[0])
completeDF = completeDF.assign(GoogleTrendsScore=trendsDF[0])
#print(completeDF)

## OBTAINS SAMPLE DATA FOR TRAINING AND TESTING MODELS
tempDF = completeDF
lm = LinearRegression()
train_x = [tempDF["ClosingPrice"][:-252]]
test_x = [tempDF["ClosingPrice"][-252:]]
train_y = [tempDF["GoogleTrendsScore"][:-252]]
test_y = [tempDF["GoogleTrendsScore"][-252:]]

## OG GRAPH
plt.scatter(train_x, train_y, color="black")
plt.xlabel("Stock Price")
plt.ylabel("Google Trend Score")
plt.show()


# LINEAR REGRESSION
lm.fit(train_x, train_y)
equation = str((np.poly1d(np.polyfit(tempDF["ClosingPrice"], tempDF["GoogleTrendsScore"], 1))))
m=equation[:equation.index("+")-2]
b=equation[equation.index("+")+2:]
json.dump(equation, codecs.open("graph.json", 'w', encoding='utf-8'), sort_keys=True, indent=4)


## LINEAR REGRESSION GRAPH

plt.scatter(train_x, train_y, color="black")
lm.fit(train_x, train_y)
plt.plot(np.unique(tempDF["ClosingPrice"]), np.poly1d(np.polyfit(tempDF["ClosingPrice"], tempDF["GoogleTrendsScore"], 1))(np.unique(tempDF["ClosingPrice"])))

plt.xlabel("Stock Price")
plt.ylabel("Google Trend Score")


plt.ylabel("Stock Price")
plt.xlabel("Google Trend Score")

print (np.unique(tempDF["ClosingPrice"]), np.poly1d(np.polyfit(tempDF["ClosingPrice"], tempDF["GoogleTrendsScore"], 1))(np.unique(tempDF["ClosingPrice"])))

plt.show()

plt.xlabel("Stock Price")
plt.ylabel("Google Trend Score")
plt.show()

print (np.unique(tempDF["ClosingPrice"]), np.poly1d(np.polyfit(tempDF["ClosingPrice"], tempDF["GoogleTrendsScore"], 1))(np.unique(tempDF["ClosingPrice"])))

plt.show()
#EQUATION
equation = (np.poly1d(np.polyfit(tempDF["ClosingPrice"], tempDF["GoogleTrendsScore"], 1)))


# TRAINING THE MODEL
lm.predict(train_x)
plt.scatter(train_x, lm.predict(train_x))
plt.xlabel("Stock Price")
plt.ylabel("Google Trend Score")
train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=0)
print(train_x.shape, test_x, train_y, test_y)