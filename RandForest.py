import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib
from sklearn import datasets

url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=demo"

data = pd.read_json(url["Time Series (Daily)"], orient="series")
print(data.head())