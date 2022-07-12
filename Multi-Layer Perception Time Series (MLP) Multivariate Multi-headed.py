# multivariate multi-headed mlp example

from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
 X, y = list(), list()
 for i in range(len(sequences)):
  # find the end of this pattern
  end_ix = i + n_steps
  # check if we are beyond the dataset
  if end_ix > len(sequences):
   break
  # collect input and output pattern data 
  seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
  X.append(seq_x)
  y.append(seq_y)
 return array(X), array(y)
 
# define input sequence
in_seq1 = array([5, 10, 15, 20, 25, 30, 35, 40, 45])
in_seq2 = array([15,25, 35, 45, 55, 65, 75, 85,95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

# choose a number of time steps
n_steps = 3

# convert into input/output
X, y = split_sequences(dataset, n_steps)
# separate input data
X1 = X[:, :, 0]
X2 = X[:, :, 1]

# first input model
visible1 = Input(shape=(n_steps,))
dense1 = Dense(100, activation='relu')(visible1)

# second input model
visible2 = Input(shape=(n_steps,))
dense2 = Dense(100, activation='relu')(visible2)

# merge input models
merge = concatenate([dense1, dense2])
output = Dense(1)(merge)
model = Model(inputs=[visible1, visible2], outputs=output)
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

# fit model
model.fit([X1, X2], y, epochs=2500, verbose=1)

# demonstrate prediction
x_input1 = array([[5, 15], [10, 25], [15, 35]])
x1 = x_input1[:, 0].reshape((1, n_steps))
x2 = x_input1[:, 1].reshape((1, n_steps))
y1 = model.predict([x1, x2], verbose=0)
print(y1)

x_input2 = array([[7, 17], [12, 27], [17, 37]])
x1 = x_input2[:, 0].reshape((1, n_steps))
x2 = x_input2[:, 1].reshape((1, n_steps))
y2 = model.predict([x1, x2], verbose=0)
print(y2)