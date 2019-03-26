import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# keras for LSTM
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint

# datatrain_set = pd.read_csv('BitMEX-OHLCV-1d.csv')
# datatrain_set.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

datatrain_set = pd.read_csv('NSE-TATAGLOBAL.csv')
datatrain_set.head()

training_set = datatrain_set.iloc[:, 1:2].values


sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []

for i in range(60, 2035):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

filepath = "model.h5"

# processing LSTM
model = Sequential()
model.add(LSTM(units = 50, return_sequences= True, input_shape= (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences= True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences= True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer='adam', loss='mean_squared_error')

# fit and save model
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x_train, y_train, epochs=2, batch_size=32, callbacks=callbacks_list)


# load the model
new_model = load_model("model.h5")
assert_allclose(model.predict(x_train),
                new_model.predict(x_train),
                1e-5)

# fit the model
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
new_model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=callbacks_list)



# predict future stock
dataset_test = pd.read_csv('BitMEX-OHLCV-1d.csv')
dataset_test.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

# dataset_test = pd.read_csv('tatatest.csv')
# dataset_test.head()

real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((datatrain_set['Open'], dataset_test['open']), axis = 0)
# dataset_total = pd.concat((datatrain_set['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
x_test = []

for i in range(60, 160):
    x_test.append(inputs[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = new_model.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'black', label = 'Bitcoin Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Bitcoin Price')
plt.title('Bitcoin Prie Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
