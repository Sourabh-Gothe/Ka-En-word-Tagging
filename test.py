import nltk
from nltk.classify import maxent
#from tkinter import *
#train = [('Nimma', 'k'),('Nimage','k'),('Naanu','k'),('Nanna','k'),('Graduates','e'),('email','e'),('resume','e'),('enjoy','e'),('com','e'),('forum','e')]
#test = ["Naavu","nimmalli","B.com","namma"]

train = [({"Nimma":'' },"kan") ,({"resume": 0}, "eng"), ({"enjoy": 0}, "eng"),({"Nimma": 0},"kan"),({"Naanu": 0},"kan"),({"Nimage": 0}, "kan"),({"nanna": 0}, "kan"),({"NANAGE": 0}, "kan"),({"nodi": 0}, "kan"),({"forum": 0}, "eng"),({"careful": 0}, "eng"),({"problem": 0}, "eng")]
test = [{"enjoy": 0}, {"resume": 0}, {"AIDS": 0},{"forum": 0}, {"careful": 0}, {"problem": 0}]
encoding = maxent.TypedMaxentFeatureEncoding.train(train, count_cutoff=3, alwayson_features=True)
classifier = maxent.MaxentClassifier.train(train, bernoulli=True, encoding=encoding, trace=0)
print classifier.classify_many(test)
