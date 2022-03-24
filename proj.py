#python
#morphology final project etc etc


import fasttext, editdistance
from fasttext import util
import numpy as np
import io


print(editdistance.distance('cat', 'bat'))

def cosine(veca, vecb):
    return (np.dot(veca, vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))

print(cosine([1, 2, 1], [1, 1, 1]))


# creates a dictionary ([str -> map object]) that maps a word to its vector representation
# by reading the first nb_of_words words of fname file (which should be in the .vec format of fasttext)
def load_vectors(fname, nb_of_words):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    i = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        i += 1
        if i >= nb_of_words:
            break
    return data

data1 = load_vectors("wiki-news-300d-1M.vec", 100)
data2 = load_vectors("wiki-news-300d-1M-subword.vec", 100)
