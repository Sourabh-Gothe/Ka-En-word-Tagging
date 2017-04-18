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
from sklearn.utils import shuffle
import pandas as pd
import requests
import json
from googletrans import Translator



def GetDataset(filename):
    fp = open(filename,"r",encoding="utf8")
    f = fp.read()
    tokens = nltk.word_tokenize(f)
    tag = 1
    #print(tokens)
    for x in tokens:
        if x == 'ENGLISH' or x =='english':
            tag = 0
            #print(tag)
        training_words.append(x)
        training_tags.append(tag)
    #print(training_tags.count(1))
    #print(training_tags.count(0))
    


def GetVectors(filename):
    fp = open(filename,"r",encoding="utf8")
    file = fp.readlines()
    #print(file)
    vocab = [line.split() for line in file]
    print(vocab)
    vectors = word2vec.Word2Vec(vocab,min_count=1,size=MAX_WORD_LENGTH,seed=1)
    #print voc_vec.syn0.shape
    for word in training_words:
        print(word)
        Dataset.append(vectors[word])
    return vectors

def GetTestData(Sentence):
    vocab = [[s] for s in Sentence.split(" ")]
    #print(vocab)
    voc_vec = word2vec.Word2Vec(vocab,min_count=1,size=MAX_WORD_LENGTH)
    X =[]
    for word in Sentence.split(" "):
        try:
            X.append(vectors[word])
            print("found"+ " "+word)
        except KeyError:
            X.append(voc_vec[word])
    return X
        

MAX_WORD_LENGTH = 3
n_classes = 2
n_dim = 3
#Training data
training_words   = []
#Target data
training_tags = []
#Converting Words into Vectors
Dataset  = []
class_labels = training_tags
 

GetDataset("our_own_dataset.txt")
vectors = GetVectors("our_own_dataset.txt")
Dataset = [x.tolist() for x in Dataset]
print(len(Dataset))
print(len(class_labels))


X_train, X_test, y_train, y_test = train_test_split( Dataset, class_labels, test_size=0.2, random_state=1 )


#Shuffling
X_train,y_train = shuffle(X_train, y_train, random_state=17)
X_test, y_test  = shuffle( X_test , y_test, random_state=17)


# Converting labels to binary vectors
trainY = to_categorical(y_train, nb_classes=n_classes)
testY = to_categorical(y_test, nb_classes=n_classes)


# Network building
net = tflearn.input_data([None, MAX_WORD_LENGTH])
#net = tflearn.embedding(net, input_dim=MAX_WORD_LENGTH, output_dim=MAX_WORD_LENGTH,trainable=True, name="EmbeddingLayer")
#net = tflearn.lstm(net, MAX_WORD_LENGTH*15)
net = tflearn.dropout(net,0.5)
net = tflearn.fully_connected(net, n_classes, activation='softmax')
net = tflearn.regression(net,learning_rate=0.1, optimizer='adam',
                         loss='categorical_crossentropy')


model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=3)

model.fit(X_train, trainY,n_epoch=100,shuffle=True, show_metric=True,
      batch_size=10)
    

while True:
    Sentence = str(input("Please enter something\n").strip())
    X_test = GetTestData(Sentence)
    X_test = [x.tolist() for x in X_test]
    predY=np.array(model.predict(X_test))
    y_pred=np.around(predY[:,0])
    print(y_pred)
    sentence_list = Sentence.split(" ")
    string = ""
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            string += sentence_list[i]+","
        print(string)
    words = ['']
    line = ""
    if string != "":
        url = "http://www.google.com/transliterate/indic?tlqt=1&langpair=en|kn&text="+string+"&&tl_app=1"
        r = requests.get(url);
        buffer = r.text
        buffer.replace(",\n]","]")
        buffer = json.loads(buffer)
        
        
        c = 0
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                sentence_list[i] =  buffer[c]['hws'][0]
                c = c +1
            
    for i in range(len(y_pred)):
        line += sentence_list[i] + " "
    print(line)
        
    """print(buffer[0]['hws'][0])
    print(buffer[1]['hws'][0])
    print(buffer[2]['hws'][0])"""
    
    
    translator = Translator()
    translations = translator.translate(line,dest='en')
    print(translations.origin, ' -> ', translations.text)
    
    
    
    
    
    
    
    
    
    
    
    
    
    



"""predY=np.array(model.predict(X_test))
print("result-->")
print(predY)
y_pred=np.around(predY[:,0])
output = np.savetxt("output.txt",y_pred)
print(y_pred)
print("Actual-->")
print(testY)
print(classification_report(y_test, y_pred) )
print(accuracy_score(y_test, y_pred))
#print(model.evaluate(X_test,testY))"""







