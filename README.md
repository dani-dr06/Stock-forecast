# Stock Forecasting

## Summary

![](https://github.com/dani-dr06/Stock-forecast/blob/main/images/stock_pred.png)


The objective of this project was to train a machine learning model, more especifically a gradient boosting model, to predict the performance of stocks. Netflix stock data was collected using yfinance and the model was trained on this data which ranged from January 2005 to December 2023, and later the model was tested with other companies as well. Sklearn's gradient boosting regressor implementation was used and the model was able to predict historical stock data for multiple companies well. However, the model did show issues when prices spiked or went above a certain threshold, seemingly around the $600-700 mark, which is definitely an area of improvement. Later on the model was used to forecast 2 months worth of data for four different stocks; they were not accurate. Finally, I trained an LSTM model using TensorFlow, which once again was accurate with historical data.

## Libraries and Dataset
The following packages were used

- Pandas
- Numpy
- Matplotlib
- yfinance
- sklearn
- TensorFlow

The yfinance package was used to obtain the stock data. Data can be downloaded into a pandas DataFrame using the following line of code:
```
df = yf.download(<stock symbol>, <start date>, <end date>)
```

For example, to obtain Amazon stock data from January 1st, 2018 to December, 2020 (note that the end date is excluded):
```
import pandas as pd
import yfinance as yf

df = yf.download('amzn', '2018-01-01' , '2020-12-01')
```

The dataframe will contain daily data with the following columns regarding the stock:
1. Open - first traded price
2. High - highest price
3. Low - lowest price
4. Close - last traded price
5. Adj Close - closing price accounting for corporate actions
6. Volume - number of shares traded

## Feature Engineering and Training

Since the target metric utilized to predict stock performance was the closing price, I decided to create features where a given day's closing price is dependent on the previous week's data; the model takes into account each of the previous seven days' closing prices. In addition, the average closing price for the previous week along with the standard deviation is also included. After defining our features and target variable, the model was trained using times series cross validation via sklearn's TimeSeriesSplit to determine which hyperparameters provided the best results. Mean absolute error and mean squared error were the metrics used to evaluate the model, and overall when it comes to predicting historical stock data the model performs very well, except when there is an increase in the closing price above a threshold found in the $600-700 mark. For the LSTM, the same process was followed, except I increased the lagged window size to 30.

## Forecasting
Since the features are dependent on the previous seven days and, obviously, with forecasting we do not have data available, for forecasting I started out by obtaining a small amount of data prior to the first date of our forecasting time period, so that it could be used to calculate the lagged features for the first few days. Then, the process consisted of calculating the features based on the predicted values by the model, and this is a process repeated for each day in our forecasting time period.

