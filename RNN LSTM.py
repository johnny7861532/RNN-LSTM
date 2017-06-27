
"""
Created on Tue Jun  6 17:06:15 2017

@author: johnnyhsieh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the trainning data
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

#Feature scaling normalize the data for more effinence learning
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#Getting the inputs and the outputs
X_train = training_set[0:1257]
Y_train = training_set[1:1258]

#reshaping
X_train = np.reshape(X_train,(1257,1,1))

# import the Keras packages and libraries

from keras.models import Sequential
from keras.layers import Dense, LSTM

#inital the RNN
regressor = Sequential()

#add the input layer and the LSTM layer
#if u got over two LSTM u need to add one more parameter return_sequences = True
#,and  remeber to remove and the next LSTM layer
regressor.add(LSTM(units = 4,activation = 'sigmoid',input_shape = (None, 1)))


#add the output layer
regressor.add(Dense(units = 4))
regressor.add(Dense(units = 2))
regressor.add(Dense(units = 1))

#compiling the RNN

regressor.compile(optimizer = 'Adam', loss = 'mean_squared_error')

#fit and train the model

regressor.fit(X_train,Y_train,batch_size=32,epochs = 200)

regressor.save_weights('RNN LSTM.h5')

#get the real stock price of 2017

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

#trying to predict the real stock price of google 2017

inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs,(20,1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualising the result

plt.plot(real_stock_price, color ='red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predict Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend
plt.show()

#getting the real stock price with google from 2012 ~ 2016
real_stock_price_train = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train = real_stock_price_train.iloc[:,1:2].values

#getting the predict stock price
#we don't need to preprocessing the data because we have done that before
predict_stock_price_train = regressor.predict(X_train)
preidct_stock_price_train = sc.inverse_transform(predict_stock_price_train)

#visuallize the result
plt.plot(real_stock_price_train, color ='red', label = 'Real Google Stock Price')
plt.plot(preidct_stock_price_train, color = 'blue', label = 'Predict Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend
plt.show()

#evaluating the RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))/800
rmse
