from __future__ import division, print_function, absolute_import
import tflearn         #For RNN Implementation
from tflearn import Evaluator
from tflearn.data_utils import to_categorical #for output class categorization
from gensim.models import word2vec  #Tool which converts words to vectors
from collections import OrderedDict #Not used
import numpy as np  #numpy operations
import token #For tokenization
import nltk #For tokenizing

def GetDataset(filename):
    fp = open(filename,"r")
    f = fp.read()
    tokens = nltk.word_tokenize(f)
    for word in tokens[0:200]:
        training_words.append(word.split("|")[0])
        training_tags.append(word.split("|")[1])


def GetVectors(filename):
    fp = open(filename,"r")
    file = fp.readlines()
    vocab = [line.split() for line in file]
    vectors = word2vec.Word2Vec(vocab,min_count=1,size=3)
    #print voc_vec.syn0.shape
    for word in training_words:
        Dataset.append(vectors[word])


#Training data
training_words   = []
#Target data
training_tags = []
GetDataset("train_data.txt")

#Converting Words into Vectors
Dataset  = []
class_labels = training_tags
GetVectors("plain_data.txt")
Dataset = [x.tolist() for x in Dataset]

tr_begin = 0
tr_end   = 160
trainX = Dataset[tr_begin:tr_end]
trainY = to_categorical(class_labels[tr_begin:tr_end], nb_classes=2)

tst_begin = 160
tst_end   = 180
testX = Dataset[tst_begin:tst_end]
testY = to_categorical(class_labels[tst_begin:tst_end], nb_classes=2)

vld_begin = 180
vld_end   = 200
validX = Dataset[vld_begin:vld_end]
validY = to_categorical(class_labels[vld_begin:vld_end], nb_classes=2)


# Network building
net = tflearn.input_data([None, 3])
print(net)
exit(0)
net = tflearn.embedding(net, input_dim=3, output_dim=3)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')


model = tflearn.DNN(net, tensorboard_verbose=3)
model.fit(trainX, trainY, validation_set=(validX, validY), show_metric=True,
          batch_size=5,n_epoch=5)


model = Evaluator(net)
print(testX)
print("Predicted --> " + str(model.predict(feed_dict={"InputData": testX}))+" \nActual --> " +str(testY))



