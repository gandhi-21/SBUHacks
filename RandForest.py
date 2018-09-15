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
with open("trends.txt") as f:
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
m = lm.coef_
b = lm.intercept_
print(m, b)

lm.fit(train_x, train_y)
m = lm.coef_.tolist()
b = lm.intercept_.tolist()
print(m, b)
npa = [].append(m).append(b)
json.dump(npa, codecs.open("graph.json", 'w', encoding='utf-8'), sort_keys=True, indent=4)


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

