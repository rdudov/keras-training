'''
Predicts sin graph.

'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, TimeDistributed, RepeatVector

import matplotlib.pyplot as plt
from pylab import *
from datetime import datetime
import time
from matplotlib.dates import  DateFormatter, WeekdayLocator, HourLocator, \
     DayLocator, MONDAY
from matplotlib.finance import candlestick_ohlc

import numpy as np
import sys

k_p_large = 5.
k_a_small = .3
step = 20
A = 100
model_name = 'candles'

X = np.linspace(0,100000,1000000)
Y = np.asarray(A + np.sin(1/k_p_large*X))

data = Y
examples = 40
y_examples = 40
s_n = .1

nb_samples = len(Y) - examples

def candle(data,start,stop):
    open = data[start]
    close = data[stop-1]
    low = min(data[start:stop])
    high = max(data[start:stop])
    return np.asarray([.05*(start+stop),open,high,low,close])

def trend(start, stop):
    if (start > stop):
        return [0, 1]
    elif (start < stop):
        return [1, 0]
    else:
        return [0, 0]

input_c_list = [np.atleast_2d(candle(data,i,i+step)) for i in xrange(0,nb_samples-step,step)]
input_c_mat = np.concatenate(input_c_list, axis=0)

'''
for i, c in enumerate(input_mat):
    if (c[1] < c[0]):
        print(i, "O", c[0], "H", c[1])
    if (c[1] < c[2]):
        print(i, "L", c[2], "H", c[1])
    if (c[1] < c[3]):
        print(i, "C", c[3], "H", c[1])
'''

steps = [-.02, -.01, -.005, -.002, -.001, -.0005, -.0001, 0, .0001, .0005, .001, .002, .005, .01, .02]

def character(data_prev, data_cur):
    rate = (data_cur-data_prev)/data_prev
    for i in range(len(steps)):
        if rate <= 0 and rate <= steps[i]:
            return i
        if rate >= 0 and rate < steps[i]:
            return i-1
    return len(steps)-1

def characterize(data):
    return [(character(data[i-1,1],data[i,1]), character(data[i-1,1],data[i,4])) for i in xrange(1,len(data))]

def measure(val_prev, character):
    return [val_prev*(1+steps[character[0]]), val_prev*(1+steps[character[1]])]

def measurize(text, start_y=0, start_x=0, step_size=step):
    res = np.zeros((len(text),5))
    val_prev = start_y
    t = start_x + .05 * step_size
    for i in xrange(0,len(text)):
        [O,C] = measure(val_prev, text[i])
        H = max(O,C)
        L = min(O,C)
        res[i] = [t, O,H,L,C]
        val_prev = O
        t += .1*step_size
    return res

text = characterize(input_c_mat)
chars = set(text)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

sentences = []
next_sentences = []

for i in range(len(text) - examples - y_examples):
    sentences.append(text[i: i + examples])
    next_sentences.append(text[i + examples:i+examples+y_examples])
print('nb sequences:', len(sentences))

X_data = np.zeros((len(sentences), examples, len(chars)), dtype=np.bool)
y_data = np.zeros((len(sentences), y_examples, len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X_data[i, t, char_indices[char]] = 1

for i, next_sentence in enumerate(next_sentences):
    for t, next_char in enumerate(next_sentence):
        y_data[i, t, char_indices[next_char]] = 1

data_new = measurize(text,A)

'''
TODO:
Make input data from text
Make reference data from text
1. Predict reference data as is
2. Predict trends
3. If doesn't work prorerly check if text can be generated from graph.
'''

'''
show_steps = step*examples + step*y_examples
fig, (ax1,ax2) = plt.subplots(2)
candlestick_ohlc(ax1, input_c_mat[:show_steps/step], width=.06*step)
ax1.plot(X[:show_steps], Y[:show_steps], label='Y')
candlestick_ohlc(ax2, data_new[:show_steps/step], width=.06*step)
ax2.plot(X[:show_steps], Y[:show_steps], label='Y')
plt.show()
'''


#model = model_from_json(open(model_name + '.json').read())


hidden = 128
model = Sequential()
model.add(LSTM(input_shape=(examples, len(chars)), output_dim=hidden))
model.add(RepeatVector(examples))
model.add(LSTM(output_dim=hidden, return_sequences=True))
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

print('Compile model ' + model_name)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])

print('Save model ' + model_name)

#Save model architecture
json_string = model.to_json()
open(model_name + '.json', 'w').write(json_string)

print('Train model ' + model_name)

#model.load_weights(model_name + '.h5')

# Train
nb_epochs = 1
for i in range(nb_epochs):
    print('Epoch', i+1, '/', nb_epochs)
    model.fit(X_data,
              y_data,
              nb_epoch=1)

    #Save weights
    model.save_weights(model_name + '.h5', overwrite=True)

print('Predicting...')

#TODO fix dimensions
generated_text = indices_char[np.argmax(model.predict(X_data[i:i+1,:,:])[0])]

generated_data = measurize(generated_text, input_c_mat[examples-1,1], examples)

print('Plotting...')

show_steps = examples + y_examples
fig, (ax1,ax2,ax3) = plt.subplots(3)
candlestick_ohlc(ax1, input_c_mat[:show_steps], width=.06*step)
ax1.plot(X[:show_steps*step], Y[:show_steps*step], label='Y')
candlestick_ohlc(ax2, data_new[:show_steps], width=.06*step)
ax2.plot(X[:show_steps*step], Y[:show_steps*step], label='Y')
candlestick_ohlc(ax3, generated_data[:y_examples], width=.06*step)
ax3.plot(X[:show_steps*step], Y[:show_steps*step], label='Y')
plt.show()
