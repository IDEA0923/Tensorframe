import numpy as np
import funcs

sentences = funcs.import_senteces("ai_loop_dataset.txt")
words = funcs.make_word_list(sentences=sentences)

hidden_layer , alfa , itr = 20 , 0.1 , 2

w0 = np.random.random(hidden_layer , len(words))
w1 = np.random.random(len(words) , hidden_layer)

one_matrix =  np.eye(hidden_layer)
def learn():
    start = np.zeros(hidden_layer)
    l1 = list()
    for i in itr :
        start = np.zeros(hidden_layer)
        l1_t = {}
        for sent in sentences: 
            
            for w in sent:
                

learn()