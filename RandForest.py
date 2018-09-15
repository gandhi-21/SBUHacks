import pandas as pd
import pytrends
from pytrends.request import TrendReq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


url = "https://api.iextrading.com/1.0/stock/AMD/chart/5y?filter=date,close"
urlInfo = "https://api.iextrading.com/1.0/stock/AMD/company"
closingprices = []
dates = []
data = pd.DataFrame(pd.read_json(url, orient="columns"))
# pd.set_option("display.max_rows", 105)
for i in data:
    dates.append(data["date"])
    closingprices.append(data["close"])

cp = list(reversed(closingprices))

pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["AMD"]
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
returnedTrend=pytrends.interest_over_time()
retTr = returnedTrend.iloc[::-1]
ratings = retTr["AMD"].tolist()

# print("----------------------------INITIAL DATA------------------------------")
# print(cp)
# print(retTr)
# print("------------------------GOOGLE TRENDS API ARRAY-----------------------")
# print(ratings)
print(len(ratings))
DATES = np.array(dates)
CLOSINGPRICES = np.array(cp)
TRENDSCORES = np.array(ratings)
DatesDF = pd.DataFrame(data=DATES.T)
pricesDF = pd.DataFrame(data=CLOSINGPRICES.T)
trendsDF = pd.DataFrame(data=TRENDSCORES.T)

completeDF = pd.DataFrame()
completeDF = completeDF.assign(Date=DATES[0])
completeDF = completeDF.assign(ClosingPrice=CLOSINGPRICES[0])
completeDF = completeDF.assign(GoogleTrendsScore=trendsDF[0])
print(completeDF)
##### THIS IS TEMPORARY, GOOGLE TRENDS API ONLY RETRIEVES 260 RECENT QUERIES
tempDF = completeDF.loc[0:260]
lm = LinearRegression()
train_x = tempDF["ClosingPrice"][:-52]
test_x = tempDF["ClosingPrice"][-52:]
train_y = tempDF["GoogleTrendsScore"][:-52]
test_y = tempDF["GoogleTrendsScore"][-52:]

lm.fit(train_x, train_y)
stock_y_pred = lm.predict(train_x)


#####






# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))