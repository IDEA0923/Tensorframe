import numpy as np

def import_senteces(file_name: str)->list :
    f =  open(file=file_name , mode="r+")
    sent =  list()
    for i in f.readlines():
        words = list()
        buff= ""
        for i1 in i :
            if(i1 != ' ' and i1 != '\n'):
                buff =  buff + i1
            elif(i1 == ' '):
                words.append(buff)
                
                buff = ""
        words.append(buff)
        sent.append(words)
    return sent



def make_word_list(sentences : list)->list:
    wordlist = list()
    for i in sentences:
        for i1 in i :
            if i1 not in wordlist :
                wordlist.append(i1)
    return wordlist

