# -*- coding: utf-8 -*-
"""
For Dataset -->  https://github.com/tflearn/tflearn/files/381249/code.and.data.zip
or take from mobile
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.

References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).

Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/

"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tensorflow.contrib import learn
import pandas
import numpy as np
#from string import replace
import tensorflow as tf
from numpy import float32
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import os

EMBEDDING_SIZE = 100
MAX_DOCUMENT_LENGTH=200
n_classes=2

def readMyDataCsv():
    X=[]
    y=[]
    dirs={0:"E:\\Projects\\CodeSwitching\\code.and.data\\code and data\\neg",
          1:"E:\\Projects\\CodeSwitching\\code.and.data\\code and data\\pos"}
    for key in dirs:
        for f in os.listdir(dirs[key]):
            filebuff=""
            for line in open(dirs[key]+"/"+f):
                filebuff+=line.strip()+" "
            X.append(filebuff)
            y.append(key)
    return X,y

X,y = readMyDataCsv()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1 )


print(len(X_train))
vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
y_train = np.array(y_train).astype(int)

X_test = np.array(list(vocab_processor.transform(X_test)))
y_test = np.array(y_test).astype(int)


n_words = len(vocab_processor.vocabulary_)


# Data preprocessing
# Sequence padding
trainX = pad_sequences(X_train, maxlen=MAX_DOCUMENT_LENGTH, value=0. , truncating='post')
testX = pad_sequences(X_test, maxlen=MAX_DOCUMENT_LENGTH, value=0., truncating='post')
# Converting labels to binary vectors
trainY = to_categorical(y_train, nb_classes=n_classes)
testY = to_categorical(y_test, nb_classes=n_classes)

print(trainX)
print(trainY)
print(testX)
# Network building
net = tflearn.input_data([None, MAX_DOCUMENT_LENGTH])
net = tflearn.embedding(net, input_dim=n_words, output_dim=EMBEDDING_SIZE,trainable=True, name="EmbeddingLayer")
net = tflearn.lstm(net, EMBEDDING_SIZE)
net = tflearn.dropout(net,0.5)
net = tflearn.fully_connected(net, n_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam',
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)

i = 0
while i != 1:
    i = i+1
    model.fit(trainX, trainY, validation_set=(testX, testY),n_epoch=1,shuffle=True, show_metric=True,
          batch_size=100)
    predY=np.array(model.predict(testX))
    print("result-->")
    print(predY)
    y_pred=np.around(predY[:,0])
    print(y_pred)
    print("Actual-->")
    print(testY)
    print(classification_report(y_test, y_pred) )
    print(accuracy_score(y_test, y_pred))
    print(model.evaluate(testX,testY))

