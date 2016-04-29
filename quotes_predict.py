'''
Predicts graph for real quotes.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, TimeDistributed, RepeatVector
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

model_name = 'quotes_predict'

examples = 40
y_examples = 20
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
    data = pd.concat((data,df),ignore_index=True)

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
next_sentences = []

for i in range(len(text) - examples - y_examples):
    sentences.append(text[i: i + examples])
    data_new[i] = measurize(text[i: i + examples + y_examples],data_mat[i,1],i)
    next_sentences.append(text[i + examples:i+examples+y_examples])
print('nb sequences:', len(sentences))

nb_samples = int(.9*len(sentences))
nb_examples = int(.1*len(sentences))

print('Making training data...')

X_train = np.zeros((nb_samples, examples, len(chars)), dtype=np.bool)
y_train = np.zeros((nb_samples, y_examples, len(chars)), dtype=np.bool)
X_test = np.zeros((nb_examples, examples, len(chars)), dtype=np.bool)
y_test = np.zeros((nb_examples, y_examples, len(chars)), dtype=np.bool)

for i in range(nb_samples):
    sentence = sentences[i]
    for t, char in enumerate(sentence):
        X_train[i, t, char_indices[char]] = 1

    next_sentence = next_sentences[i]
    for t, next_char in enumerate(next_sentence):
        y_train[i, t, char_indices[next_char]] = 1
        
print('Making testing data...')

for i in range(nb_examples):
    sentence = sentences[nb_samples+i-1]
    for t, char in enumerate(sentence):
        X_test[i, t, char_indices[char]] = 1

    next_sentence = next_sentences[nb_samples+i-1]
    for t, next_char in enumerate(next_sentence):
        y_test[i, t, char_indices[next_char]] = 1

'''
show_steps = examples + y_examples
fig, (ax1,ax2) = plt.subplots(2)
candlestick_ohlc(ax1, data_mat[:show_steps], width=.6*step, colorup='grey', colordown='k')
candlestick_ohlc(ax2, data_new[0,:], width=.6*step, colorup='grey', colordown='k')
plt.show()
'''

#model = model_from_json(open(model_name + '.json').read())


hidden = 128
model = Sequential()
model.add(LSTM(input_shape=(examples, len(chars)), output_dim=hidden))
model.add(RepeatVector(y_examples))
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

def predict_and_plot(offset=0, show=False, save=True, path='plots/' + model_name + '.png'):
    
    print('Predicting...')

    predicted_data = model.predict(X_test[offset:offset+1,:,:])[0]
    generated_text = [indices_char[np.argmax(predicted_data[i,:])] for i in xrange(y_examples)]
    generated_data = measurize(generated_text, data_mat[examples+offset-1,1], nb_samples+x_start+(examples+offset)*step)


    print('Plotting...')

    show_steps = examples + y_examples
    fig, (ax1,ax2) = plt.subplots(2)
    candlestick_ohlc(ax1, data_mat[nb_samples+offset:nb_samples+offset+show_steps], width=.6*step, colorup='grey', colordown='k')
    candlestick_ohlc(ax1, generated_data[:y_examples], width=.6*step*step,alpha=.5, colorup='r', colordown='k')
    candlestick_ohlc(ax2, data_new[nb_samples+offset], width=.6*step, colorup='grey', colordown='k')
    candlestick_ohlc(ax2, generated_data[:y_examples], width=.6*step*step,alpha=.5, colorup='r', colordown='k')
    if show:
        plt.show()
    if save:
        print('Saving plot to', path)
        fig.savefig(path)   # save the figure to file
        plt.close(fig)    # close the figure

# Train
nb_epochs = 30
for i in range(nb_epochs):
    print('Epoch', i+1, '/', nb_epochs)
    model.fit(X_train,
              y_train,
              nb_epoch=1)

    #Save weights
    model.save_weights(model_name + str(i) + '.h5', overwrite=True)

    for j in range(3):
        offset = np.random.randint(len(X_test)-1)
        predict_and_plot(offset=offset, path='plots/' + model_name + '_ep'+str(i) + '_plt' + str(j) + '.png')



predict_and_plot(offset = 0, show=True,save=False)