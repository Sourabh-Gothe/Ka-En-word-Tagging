import nltk
#from tkinter import *
from nltk.corpus import stopwords



def read(event):
    fp = open("","r")
    f = fp.read()
    t2 = nltk.word_tokenize(f)
    prt = nltk.PorterStemmer()
    #t1 = [prt.stem(t) for t in t1]
    t3 = nltk.pos_tag(t2)
    stop = stopwords.words('english')
    stop.append("By")
    list1 = []
    for (word, tag) in t3:
        if tag[:3] == "NNP" or tag[:3] =="NN" and word not in stop:
            list1.append(word)
    op = open("output.txt","w")
    list2 = []
    g.delete(1.0,END)
    g1.delete(1.0,END)
    g1.insert(END,val2.get())
    for word in list1:
        if word.startswith(val2.get()) and word not in list2 and word not in stop:
            list2.append(word)
            g.insert(END,word)
            g.insert(END,"\n")
            op.write(val2.get()+"\n")
            op.write(word+"\n")
    op.close()
    fp.close()    
    
            

# mainloop()
