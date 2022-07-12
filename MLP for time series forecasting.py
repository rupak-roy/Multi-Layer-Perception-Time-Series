#MLP for time series forecasting

from pandas import read_csv
dataframe = read_csv('covid19processed_data.csv', usecols=[0], engine='python')
dataset = dataframe.values

#converting multidimensional array to single dim
data = dataset.flatten() 
raw_seq = data.tolist()

#split a univariate sequence into samples
def split_sequence(sequence, n_steps):
 X, y = list(), list()
 for i in range(len(sequence)):
  # find the end of this pattern
  end_ix = i + n_steps
  # check if we are beyond the sequence
  if end_ix > len(sequence)-1:
   break
  #input and output sample
  seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
  X.append(seq_x)
  y.append(seq_y)
 return array(X), array(y)

from numpy import array
from keras.models import Sequential
from keras.layers import Dense

#choose a number of time steps
n_steps = 3
#split into samples
X, y = split_sequence(raw_seq, n_steps)

#define model
model = Sequential()
model.add(Dense(150, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics =['accuracy'])

#fit model
model.fit(X, y, epochs=3000, verbose=1)

#prediction for 156thday
x_input = array([13758429,14377942,15046478])
x_input = x_input.reshape((1, n_steps))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#prediction for 157th day
x_input = array([14377942,15046478,15733087])
x_input = x_input.reshape((1, n_steps))
yhat = model.predict(x_input, verbose=1)
print(yhat)

#prediction for 158th day
x_input = array([15046478,15733087,16455969])
x_input = x_input.reshape((1, n_steps))
yhat = model.predict(x_input, verbose=1)
print(yhat)