from __future__ import division, print_function, absolute_import
import tflearn         #For RNN Implementation
from tflearn import Evaluator
from tflearn.data_utils import to_categorical #for output class categorization
from gensim.models import word2vec  #Tool which converts words to vectors
from collections import OrderedDict #Not used
import numpy as np  #numpy operations
import token #For tokenization
import nltk #For tokenizing
from docutils.languages import da
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def GetDataset(filename,tag):
    fp = open(filename,"r",encoding="utf8")
    f = fp.read()
    tokens = nltk.word_tokenize(f)
    for word in tokens:
        training_words.append(word)
        training_tags.append(tag)


def GetVectors(filename):
    fp = open(filename,"r",encoding="utf8")
    file = fp.readlines()
    vocab = [line.split() for line in file]
    vectors = word2vec.Word2Vec(vocab,min_count=1,size=MAX_WORD_LENGTH,seed=1)
    #print voc_vec.syn0.shape
    for word in training_words:
        Dataset.append(vectors[word])


MAX_WORD_LENGTH = 10
n_classes = 2
n_dim = 3
#Training data
training_words   = []
#Target data
training_tags = []
#Converting Words into Vectors
Dataset  = []
class_labels = training_tags


GetDataset("unlabeled_english_words.txt",0)
GetVectors("unlabeled_english_words.txt")
training_words   = []
GetDataset("unlabeled_kannada_words.txt",1)
GetVectors("unlabeled_kannada_words.txt")
Dataset = [x.tolist() for x in Dataset]



X_train, X_test, y_train, y_test = train_test_split( Dataset, class_labels, test_size=0.2, random_state=1 )
# Converting labels to binary vectors
trainY = to_categorical(y_train, nb_classes=n_classes)
testY = to_categorical(y_test, nb_classes=n_classes)


# Network building
net = tflearn.input_data([None, MAX_WORD_LENGTH])
net = tflearn.embedding(net, input_dim=MAX_WORD_LENGTH, output_dim=MAX_WORD_LENGTH,trainable=True, name="EmbeddingLayer")
net = tflearn.lstm(net, MAX_WORD_LENGTH*15)
net = tflearn.dropout(net,0.5)
net = tflearn.fully_connected(net, n_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam',
                         loss='categorical_crossentropy')


model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)

i = 0
while i != 1:
    i = i+1
    model.fit(X_train, trainY, validation_set=(X_test, testY),n_epoch=1,shuffle=True, show_metric=True,
          batch_size=10)
    predY=np.array(model.predict(X_test))
    print("result-->")
    print(predY)
    y_pred=np.around(predY[:,0])
    print(y_pred)
    print("Actual-->")
    print(testY)
    print(classification_report(y_test, y_pred) )
    print(accuracy_score(y_test, y_pred))
    #print(model.evaluate(X_test,testY))







