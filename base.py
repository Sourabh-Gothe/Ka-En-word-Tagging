import nltk
#from tkinter import *
from nltk.corpus import stopwords



def read(event):
    fp = open(val1.get(),"r")
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
        
#root = Tk()
#root.configure(background='black')
#root.geometry('1400x1030')
#lable = Label(root,text="ARTICLE INDEXING\n",font='Anton -75 bold',fg="blue",bg='black')
#lable.pack()
#lable1 = Label(root,text="ENTER THE FILE NAME\n",font='Arial -15 bold',fg="blue",bg='black')
#lable1.pack()
#val1=StringVar()
#Text1 = Entry(textvariable=val1)
#Text1.pack()
#label2 = Label(root,text="\n\nENTER THE ALPHABET\n",font='Arial -15 bold',fg="white",bg='black')
#label2.pack()
#val2 = StringVar()
#Text2 = Entry(textvariable=val2)
#Text2.pack()
#b1 = Button(root,text="CLICK\n",font='Arial -15 bold',fg='white',bg="black")
# b1.bind('<Button-1>',read)
# b1.pack()
# g1 = Text(width="4",height="1",font='Arial -24 bold',fg='white',bg="black")
# g1.pack()
# g = Text(width="40",height="10",font='Arial -15 bold',fg='white',bg="black")
# g.pack()
# mainloop()
