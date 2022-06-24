import pandas as pd
from pandas import concat, to_datetime
import numpy as np
from numpy import concatenate
import utils
import os
import sys
import datetime
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM 

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

print("H portra kollhse")
try:
    renewable = pd.read_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'renewables.pkl')
    print('Pickles retrieved')
except:
    temp = utils.load_data()
    sources = utils.fill_nan(temp[0])
    demand = utils.fill_nan(temp[1])
    # print(sources)
    # print(demand)
    # According to the U.S Energy Information Administration
    renewable = sources[['Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Large hydro']].copy()
    renewable['Total'] = renewable.sum(axis = 1)
    renewable = renewable.join(demand['Current demand'])
    renewable['Required'] = renewable['Current demand'] - renewable['Total']
    renewable['Datetime'] = renewable.index
    renewable[['DayofYear', 'Hour', 'Minute']] = np.nan
    # Convert datetime format to day of year (1-365), hours and minutes
    print("Formating data, this may take a while...")
    for idx, row in renewable.iterrows():
        renewable.loc[idx, 'DayofYear'] = int(row['Datetime'].timetuple().tm_yday)
        renewable.loc[idx, 'Hour'] = int(row['Datetime'].time().hour)
        renewable.loc[idx, 'Minute'] = int(row['Datetime'].time().minute)
    renewable.to_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'renewables.pkl')
    print("Data Pickled")

# oold the index of the first day of the final year (first row of test data)
test_year = renewable.index.get_loc(datetime.datetime(2021, 1, 1, 0, 0 ,0))
# drop the datetime column as it is useless
renewable.drop(renewable.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10]], axis=1, inplace=True)
# renewable.drop(renewable.columns[10], axis=1, inplace=True)
values = renewable.values
# ensure data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame for supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# vars: Solar-1 Wind-2 Geothermal-3 Biomass-4 Biogas-5 S_hydro-6 L_hydro-7 Total-8 Demand-9 Required-10 Day-11 Hour-12 Minute-13
# drop columns that we don't want to predict (everything but 'Required')
reframed.drop(reframed.columns[[5, 6, 7]], axis=1, inplace=True)
# reframed.drop(reframed.columns[[13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25]], axis=1, inplace=True)

print(reframed)
# split into train and test sets
values = reframed.values
# number of total five minute intervals
train = values[:test_year, :]
test = values[test_year:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

try:
    model = load_model('dataset' + os.path.sep + 'neuralnet.model')
except:
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    model.save('dataset' + os.path.sep + 'neuralnet.model')

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
print(inv_y)
print(renewable['2021-01-01':])
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)