from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector
from keras.layers import recurrent
from keras.models import model_from_json
import numpy as np
from six.moves import range


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


MODEL_NAME = 'add_rnn'

DIGITS = 3
INVERT = True
LAYERS = 1
MAXLEN = DIGITS + 1 + DIGITS

chars = '0123456789+ '
ctable = CharacterTable(chars, MAXLEN)

model = model_from_json(open(MODEL_NAME + '.json').read())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights(MODEL_NAME + '.h5')

def add(a, b):
    # Pad the data with spaces such that it is always MAXLEN
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # Answers can be of maximum size DIGITS + 1
    ans += ' ' * (DIGITS + 1 - len(ans))
    if INVERT:
        query = query[::-1]

    X = np.array([ctable.encode(query, maxlen=MAXLEN)])
    y = np.array([ctable.encode(ans, maxlen=DIGITS + 1)])

    preds = model.predict_classes(X, verbose=0)
    q = ctable.decode(X[0])
    correct = ctable.decode(y[0])
    guess = ctable.decode(preds[0], calc_argmax=False)
    print('Q', q[::-1] if INVERT else q)
    print('T', correct)
    print(colors.ok + 'OK' + colors.close if correct == guess else colors.fail + 'FAIL' + colors.close, guess)
    print('---')