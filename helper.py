import numpy as np


class RecurrentNeuralNetwork:

    def __init__(self, input, output, recurrences, expected_output, learning_rate):
        # initial input
        self.x = np.zeros(input)
        # input size
        self.input = input
        # expected output
        self.y = np.zeros(output)
        # output size
        self.output = output
        # weight matrix
        self.w = np.random.random((output, output))
        # matrix used in RMSprop in order to decay the learning rate
        self.G = np.zeros_like(self.w)
        # length of the recurrent network
        self.recurrences = recurrences
        # learning rate
        self.learning_rate = learning_rate
        # array for storing inputs
        self.ia = np.zeros((recurrences + 1, input))
        # array for storing cell states
        self.ca = np.zeros((recurrences + 1, output))
        # array for storing outputs
        self.oa = np.zeros((recurrences + 1, output))
        # array for storing hidden states
        self.ha = np.zeros((recurrences + 1, output))
        # forget gate
        self.af = np.zeros((recurrences + 1, output))
        # input gate
        self.ai = np.zeros((recurrences + 1, output))
        # cell state
        self.ac = np.zeros((recurrences + 1, output))
        # output gate
        self.ao = np.zeros((recurrences + 1, output))
        # array of expected output values
        self.expected_output = np.vstack((np.zeros(expected_output.shape[0]), expected_output.T))
        # declare LSTM cell
        self.LSTM = LSTM(input, output, recurrences, learning_rate)

    # sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

        # Forward Propagation

    def forwardProp(self):
        for i in range(1, self.recurrences + 1):
            self.LSTM.x = np.hstack((self.ha[i - 1], self.x))
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            # store cell state from the forward propagation
            self.ca[i] = cs  # cell state
            self.ha[i] = hs  # hidden state
            self.af[i] = f  # forget state
            self.ai[i] = inp  # inpute gate
            self.ac[i] = c  # cell state
            self.ao[i] = o  # output gate
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))  # activate the weight*input
            self.x = self.expected_output[i - 1]
        return self.oa

    # Back propagation
    def backProp(self):
        totalError = 0
        # cell state
        dfcs = np.zeros(self.output)
        # hidden state,
        dfhs = np.zeros(self.output)
        # weight matrix
        tu = np.zeros((self.output, self.output))
        # forget gate
        tfu = np.zeros((self.output, self.input + self.output))
        # input gate
        tiu = np.zeros((self.output, self.input + self.output))
        # cell unit
        tcu = np.zeros((self.output, self.input + self.output))
        # output gate
        tou = np.zeros((self.output, self.input + self.output))
        for i in range(self.recurrences, -1, -1):
            error = self.oa[i] - self.expected_output[i]
            tu += np.dot(np.atleast_2d(error * self.dsigmoid(self.oa[i])), np.atleast_2d(self.ha[i]).T)
            error = np.dot(error, self.w)
            self.LSTM.x = np.hstack((self.ha[i - 1], self.ia[i]))
            self.LSTM.cs = self.ca[i]
            fu, iu, cu, ou, dfcs, dfhs = self.LSTM.backProp(error, self.ca[i - 1], self.af[i], self.ai[i], self.ac[i],
                                                            self.ao[i], dfcs, dfhs)
            totalError += np.sum(error)
            # forget gate
            tfu += fu
            # input gate
            tiu += iu
            # cell state
            tcu += cu
            # output gate
            tou += ou
        self.LSTM.update(tfu / self.recurrences, tiu / self.recurrences, tcu / self.recurrences, tou / self.recurrences)
        self.update(tu / self.recurrences)
        return totalError

    def update(self, u):
        self.G = 0.95 * self.G + 0.1 * u ** 2
        self.w -= self.learning_rate / np.sqrt(self.G + 1e-8) * u
        return

    def sample(self):
        for i in range(1, self.recurrences + 1):
            self.LSTM.x = np.hstack((self.ha[i - 1], self.x))
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            maxI = np.argmax(self.x)
            self.x = np.zeros_like(self.x)
            self.x[maxI] = 1
            self.ia[i] = self.x
            # store cell states
            self.ca[i] = cs
            # store hidden state
            self.ha[i] = hs
            # forget gate
            self.af[i] = f
            # input gate
            self.ai[i] = inp
            # cell state
            self.ac[i] = c
            # output gate
            self.ao[i] = o
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            maxI = np.argmax(self.oa[i])
            newX = np.zeros_like(self.x)
            newX[maxI] = 1
            self.x = newX

        return self.oa


class LSTM:
    # LSTM cell (input, output, amount of recurrence, learning rate)
    def __init__(self, input, output, recurrences, learning_rate):
        # input size
        self.x = np.zeros(input + output)
        # input size
        self.input = input + output
        # output
        self.y = np.zeros(output)
        # output size
        self.output = output
        # cell state intialized as size of prediction
        self.cs = np.zeros(output)
        # how often to perform recurrence
        self.recurrences = recurrences
        # balance the rate of training (learning rate)
        self.learning_rate = learning_rate
        # init weight matrices for our gates
        # forget gate
        self.f = np.random.random((output, input + output))
        # input gate
        self.i = np.random.random((output, input + output))
        # cell state
        self.c = np.random.random((output, input + output))
        # output gate
        self.o = np.random.random((output, input + output))
        # forget gate gradient
        self.Gf = np.zeros_like(self.f)
        # input gate gradient
        self.Gi = np.zeros_like(self.i)
        # cell state gradient
        self.Gc = np.zeros_like(self.c)
        # output gate gradient
        self.Go = np.zeros_like(self.o)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tangent(self, x):
        return np.tanh(x)

    def dtangent(self, x):
        return 1 - np.tanh(x) ** 2

    def forwardProp(self):
        f = self.sigmoid(np.dot(self.f, self.x))
        self.cs *= f
        i = self.sigmoid(np.dot(self.i, self.x))
        c = self.tangent(np.dot(self.c, self.x))
        self.cs += i * c
        o = self.sigmoid(np.dot(self.o, self.x))
        self.y = o * self.tangent(self.cs)
        return self.cs, self.y, f, i, c, o

    def backProp(self, e, pcs, f, i, c, o, dfcs, dfhs):
        e = np.clip(e + dfhs, -6, 6)

        do = self.tangent(self.cs) * e

        ou = np.dot(np.atleast_2d(do * self.dtangent(o)).T, np.atleast_2d(self.x))

        dcs = np.clip(e * o * self.dtangent(self.cs) + dfcs, -6, 6)

        dc = dcs * i

        cu = np.dot(np.atleast_2d(dc * self.dtangent(c)).T, np.atleast_2d(self.x))

        di = dcs * c

        iu = np.dot(np.atleast_2d(di * self.dsigmoid(i)).T, np.atleast_2d(self.x))

        df = dcs * pcs

        fu = np.dot(np.atleast_2d(df * self.dsigmoid(f)).T, np.atleast_2d(self.x))

        dpcs = dcs * f

        dphs = np.dot(dc, self.c)[:self.output] + np.dot(do, self.o)[:self.output] + np.dot(di, self.i)[
                                                                                     :self.output] + np.dot(df, self.f)[
                                                                                                     :self.output]

        return fu, iu, cu, ou, dpcs, dphs

    def update(self, fu, iu, cu, ou):
        # Update forget, input, cell, and output gradients
        self.Gf = 0.9 * self.Gf + 0.1 * fu ** 2
        self.Gi = 0.9 * self.Gi + 0.1 * iu ** 2
        self.Gc = 0.9 * self.Gc + 0.1 * cu ** 2
        self.Go = 0.9 * self.Go + 0.1 * ou ** 2

        # Update our gates using our gradients
        self.f -= self.learning_rate / np.sqrt(self.Gf + 1e-8) * fu
        self.i -= self.learning_rate / np.sqrt(self.Gi + 1e-8) * iu
        self.c -= self.learning_rate / np.sqrt(self.Gc + 1e-8) * cu
        self.o -= self.learning_rate / np.sqrt(self.Go + 1e-8) * ou
        return

from pandas_datareader.data import DataReader
import yfinance as yf
import pandas as pd
# yf.pdr_override()
from datetime import datetime
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)
iterations=1000
learningRate=0.1

df = pd.read_csv('AAPL.csv')
data = df.filter(['Close'])
dataset = data.values

training_data_len = int(np.ceil( len(dataset) * .95 ))
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
# GET CELLS
training_data_len = int(np.ceil( len(dataset) * .95 ))
train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []
## 60% train -> 40% test
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# y_train = np.reshape(y_train, (y_train.shape[0], 1, 0))

#Initialize the RNN using our hyperparameters and data
RNN = RecurrentNeuralNetwork(len(x_train), len(y_train), len(train_data), y_train, learningRate)
# for i in range(len(x_train)):
#     x = x_train[:, i].reshape(-1, 1)
#     y=y_train[i]
#     h_prev, c_prev, y_pred = RNN.forwardProp()
#     loss = (y_pred - y) ** 2
#     grad_output = 2 * (y_pred - y)
#     RNN.backward(x, h_prev, c_prev, grad_output)
#     RNN.update(learning_rate)
#training time!
for i in range(1, len(train_data)):
    #Predict the next data
    RNN.forwardProp()
    #update all our weights using our error
    error = RNN.backProp()
    #For a given error threshold
    print("Reporting error on iteration ", i, ": ", error)
    if error > -10 and error < 10 or i % 10 == 0:
        #We provide a seed data
        seed = np.zeros_like(RNN.x)
        maxI = np.argmax(np.random.random(RNN.x.shape))
        seed[maxI] = 1
        RNN.x = seed
        #and predict the upcoming one
        output = RNN.sample()
        print(output)
        #finally, we store it to disk
        print(output, data)
        print("Done Writing")