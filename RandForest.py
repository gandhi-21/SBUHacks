import pandas as pd
import pytrends
from pytrends.request import TrendReq
import numpy as np
import matplotlib.pyplot as plt
import urllib


stock = "AMD"

url = "https://api.iextrading.com/1.0/stock/" + stock + "/chart/5y?filter=date,close"
urlInfo = "https://api.iextrading.com/1.0/stock/" + stock + "/company"
closingprices = []
dates = []
data = pd.DataFrame(pd.read_json(url, orient="columns"))
# pd.set_option("display.max_rows", 105)
for i in data:
    dates.append(data["date"])
    closingprices.append(data["close"])

cp = list(reversed(closingprices))

pytrends = TrendReq(hl='en-US', tz=360)
kw_list = [stock] 
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
returnedTrend=pytrends.interest_over_time()
retTr = returnedTrend.iloc[::-1]
ratings = retTr[stock].tolist()

print("----------------------------INITIAL DATA------------------------------")
print(cp)
print(retTr)
print("------------------------GOOGLE TRENDS API ARRAY-----------------------")
print(ratings)



# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))