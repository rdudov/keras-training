'''
Predicts trends in sin graph.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, TimeDistributed, RepeatVector
from keras.models import model_from_json

import matplotlib.pyplot as plt
from pylab import *
from datetime import datetime
import time
from matplotlib.dates import  DateFormatter, WeekdayLocator, HourLocator, \
     DayLocator, MONDAY
from matplotlib.finance import candlestick_ohlc
import pandas as pd
import numpy as np
import sys

model_name = 'quotes_trend'

examples = 40
y_examples = 10
step = 1
x_start = 0

print('Loading quotes...')

data_files = ['']*5

data_files[0] = u'C:\YandexDisk\Документы\Trading\котировки\SBER_20110429_20120428_1M.txt'
data_files[1] = u'C:\YandexDisk\Документы\Trading\котировки\SBER_20120429_20130428_1M.txt'
data_files[2] = u'C:\YandexDisk\Документы\Trading\котировки\SBER_20130429_20140428_1M.txt'
data_files[3] = u'C:\YandexDisk\Документы\Trading\котировки\SBER_20140429_20150428_1M.txt'
data_files[4] = u'C:\YandexDisk\Документы\Trading\котировки\SBER_20150429_20160427_1M.txt'

data = pd.DataFrame()
for i in range(len(data_files)):
    with open(data_files[i], 'rb') as f:
        df = pd.read_csv(f, sep = ',', header = 0, index_col=False)
    data = pd.concat((data,df))

data_mat = np.zeros((len(data),5))

data_mat[:,0] = data.index
data_mat[:,1:] = data[['<OPEN>','<HIGH>','<LOW>','<CLOSE>']].values

#data_mat = data_mat[:10000]

print('Characterize data...')

steps = [-.02, -.01, -.005, -.002, -.001, -.0005, -.0001, 0, .0001, .0005, .001, .002, .005, .01, .02]
'''
0..7 <= 0
7..14 >= 0
'''

steps_next = [-.01, -.005, -.002, -.001, 0, -.001, .002, .005, .01]

def character(data_prev, data_cur, steps = steps):
    rate = (data_cur-data_prev)/data_prev
    for i in range(len(steps)):
        if rate <= 0 and rate <= steps[i]:
            return i
        if rate >= 0 and rate < steps[i]:
            return i-1
    return len(steps)-1

def characterize(data):
    return [(character(data[i-1,1],data[i,1]), character(data[i-1,1],data[i,4])) for i in xrange(1,len(data))]

chars = set()
for i in xrange(len(steps)):
    for j in xrange(len(steps)):
        chars.add((i,j))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def measure(val_prev, character):
    return [val_prev*(1+steps[character[0]]), val_prev*(1+steps[character[1]])]

def measurize(text, start_y=0, start_x=x_start, step_size=step):
    res = np.zeros((len(text),5))
    val_prev = start_y
    t = start_x + int(.5 * step_size)
    for i in xrange(0,len(text)):
        [O,C] = measure(val_prev, text[i])
        H = max(O,C)
        L = min(O,C)
        res[i] = [t, O,H,L,C]
        val_prev = O
        t += step_size
    return res

text = characterize(data_mat)
data_new = np.zeros((len(text) - examples - y_examples,examples+y_examples,5))

print('Making sentences...')

sentences = []
next_chars = []

for i in range(len(text) - examples - y_examples):
    sentences.append(text[i: i + examples])
    data_new[i] = measurize(text[i: i + examples + y_examples],data_mat[i,1],i)
    next_chars.append(character(data_mat[i + examples,4],data_mat[i+examples+y_examples,4],steps_next))
print('nb sequences:', len(sentences))

nb_samples = int(.9*len(sentences))
nb_examples = int(.1*len(sentences))

print('Making training data...')

X_train = np.zeros((nb_samples, examples, len(chars)), dtype=np.bool)
y_train = np.zeros((nb_samples, len(steps_next)), dtype=np.bool)
X_test = np.zeros((nb_examples, examples, len(chars)), dtype=np.bool)
y_test = np.zeros((nb_examples, len(steps_next)), dtype=np.bool)

for i in range(nb_samples):
    sentence = sentences[i]
    for t, char in enumerate(sentence):
        X_train[i, t, char_indices[char]] = 1
        
print('Making testing data...')

for i in range(nb_examples):
    sentence = sentences[nb_samples+i-1]
    for t, char in enumerate(sentence):
        X_test[i, t, char_indices[char]] = 1
        
print('Making reference data...')

data_next_new = np.zeros((len(next_chars),5))
t = step * (examples + .5 * y_examples)
for i in range(nb_samples):
    next_char = next_chars[i]
    y_train[i, next_char] = 1
    o = data_mat[i + examples,4]
    c = o*(1 + steps_next[next_char])
    h = max(o,c)
    l = min(o,c)
    data_next_new[i] = [t,o,h,l,c]
    t += step

'''
show_steps = examples + y_examples
fig, (ax1,ax2) = plt.subplots(2)
candlestick_ohlc(ax1, data_mat[:show_steps], width=.6*step)
candlestick_ohlc(ax1, data_next_new[:1], width=y_examples*step)
candlestick_ohlc(ax2, data_new[0,:show_steps], width=.6*step)
candlestick_ohlc(ax2, data_next_new[:1], width=y_examples*step)
plt.show()
'''


#model = model_from_json(open(model_name + '.json').read())


hidden = 128
model = Sequential()
model.add(LSTM(input_shape=(examples, len(chars)), output_dim=hidden, return_sequences=True))
model.add(LSTM(hidden))
model.add(Dense(len(steps_next)))
model.add(Activation('sigmoid'))

print('Compile model ' + model_name)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])

print('Save model ' + model_name)

#Save model architecture
json_string = model.to_json()
open(model_name + '.json', 'w').write(json_string)

print('Train model ' + model_name)

#model.load_weights(model_name + '.h5')

# Train
nb_epochs = 30
for i in range(nb_epochs):
    print('Epoch', i+1, '/', nb_epochs)
    model.fit(X_train,
              y_train,
              nb_epoch=1)

    #Save weights
    model.save_weights(model_name + '.h5', overwrite=True)

print('Predicting...')

offset = 0

predicted_data = model.predict(X_test[offset:offset+1,:,:])[0]
generated_char = np.argmax(predicted_data)
t = x_start + step * (nb_samples + examples + offset + int(.5 * y_examples))
o = data_mat[nb_samples + offset + examples - 1,4]
c = o*(1 + steps_next[generated_char])
h = max(o,c)
l = min(o,c)
generated_data = [t,o,h,l,c]

print('Plotting...')

show_steps = examples + y_examples
fig, (ax1,ax2) = plt.subplots(2)
candlestick_ohlc(ax1, data_mat[nb_samples+offset:nb_samples+offset+show_steps], width=.6*step)
candlestick_ohlc(ax1, [generated_data], width=y_examples*step)
candlestick_ohlc(ax2, data_new[nb_samples+offset], width=.6*step)
candlestick_ohlc(ax2, [generated_data], width=y_examples*step)
plt.show()