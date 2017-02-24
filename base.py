from gensim.models import word2vec
from collections import OrderedDict
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score

import nltk
import token

def read(filename):
    fp = open(filename,"r")
    f = fp.readlines()  
    vocab = [s.encode('utf-8').split() for s in f]
    #print vocab
    voc_vec = word2vec.Word2Vec(vocab,min_count=1,size=4)
    #print voc_vec.syn0.shape
    #print type(voc_vec['yav'])
    #Openning data file
    fp.close()
    fp = open("test_data.txt","r")
    f = fp.read()
    tokens = nltk.word_tokenize(f)
    D = OrderedDict()
    sentences = []
    #print len(tokens)
    for word in tokens[0:200]:
        D[word.split("|")[0]] = word.split("|")[1]
        sentences.append(word.split("|")[0])
    #print D
    
    train_data = []
    
    for key in D:
        l = voc_vec[key]
        x = {}
        x['a'] =  l[0]
        x['b'] =  l[1]
        x['c'] =  l[2]
        x['d'] =  l[3]
        train_data.append((x,D[key]))
    classif = SklearnClassifier(BernoulliNB()).train(train_data)
    #print train_data
    
    
    test_data = []
    D2 = OrderedDict()
    for word in tokens[200:300]:
        D2[word.split("|")[0]] = word.split("|")[1]
    expected_list = []
    for key in D2:
        l = voc_vec[key]
        x = {}
        x['a'] =  l[0]
        x['b'] =  l[1]
        x['c'] =  l[2]
        x['d'] =  l[3]
        test_data.append(x)
        expected_list.append(D2[key])
    predicted =  classif.classify_many(test_data)
    print len(predicted)
    print len(expected_list)
    print accuracy_score(expected_list, predicted ,normalize=False)


test =  read("plain_data.txt")
