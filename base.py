from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from gensim.models import word2vec
from collections import OrderedDict
import numpy as np
import token

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import nltk

def GetTrainData(filename):
    fp = open(filename,"r")
    f = fp.read()
    tokens = nltk.word_tokenize(f)
    for word in tokens[0:200]:
        training_words.append(word.split("|")[0])
        pre_defined_tags.append(word.split("|")[1])




def GetVectors(filename):
    fp = open(filename,"r")
    file = fp.readlines()
    vocab = [line.split() for line in file]
    vectors = word2vec.Word2Vec(vocab,min_count=1,size=3)
    #print voc_vec.syn0.shape
    for word in training_words:
        trainData.append(vectors[word])



#Training data
training_words   = []
#Target data
pre_defined_tags = []
GetTrainData("train_data.txt")

#Converting Words into Vectors
trainData  = []
TargetData = pre_defined_tags
GetVectors("plain_data.txt")

testData = trainData[150:200]
testTargetData = TargetData[150:200]

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainData[:150], maxlen=100, value=0.)
testX = pad_sequences(testData, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(TargetData, nb_classes=2)
testY = to_categorical(testTargetData, nb_classes=2)


# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=3)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=5)
print(testX)
print(model.predict(testX))


