#multiple Parallel input multi-step mlp

from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense

#split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
 X, y = list(), list()
 for i in range(len(sequences)):
  #find the end of this pattern
  end_ix = i + n_steps_in
  out_end_ix = end_ix + n_steps_out
  #check if we are beyond the dataset
  if out_end_ix > len(sequences):
   break
  #collect input and output pattern data 
  seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
  X.append(seq_x)
  y.append(seq_y)
 return array(X), array(y)
 
#define input sequence
in_seq1 = array([5, 10, 15, 20, 25, 30, 35, 40, 45])
in_seq2 = array([15,25, 35, 45, 55, 65, 75, 85,95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

#convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

#horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

#choose a number of time steps
n_steps_in, n_steps_out = 3, 2
#convert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
#flatten input
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))

#flatten output
n_output = y.shape[1] * y.shape[2]
y = y.reshape((y.shape[0], n_output))

#define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse',metrics=["accuracy"])
#fit model
model.fit(X, y, epochs=3100, verbose=1)

#prediction
x_input1 = array([[35, 75,110], [40, 85,125], [45, 95,140]])
x_input1= x_input1.reshape((1, n_input))
y1 = model.predict(x_input1, verbose=1)
print(y1)