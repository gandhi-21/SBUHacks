import pandas as pd
import pytrends
from pytrends.request import TrendReq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split


url = "https://api.iextrading.com/1.0/stock/AMD/chart/1y?filter=date,high,low,open,close"
urlInfo = "https://api.iextrading.com/1.0/stock/amd/company"
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
print(returnedTrend)

# print(interestOverTime.head())

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))