#python
#morphology final project etc etc


import fasttext, editdistance, random
from fasttext import util
import numpy as np
import io

def cosine(veca, vecb):
    return (np.dot(veca, vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))


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

def filter(word):
    return True

# randomly samples n word vectors from two dictionaries of words to vectors,
# one with sub-word info, the other without and outputs two dictonaries containing only the samples
# the filter function passed as a parameter must output true or false according to whether we want to keep the word or not
def sample_words(with_sub_word, no_sub_word, n, filter):
    sample_sw = {}
    sample_no_sw = {}
    while len(sample_no_sw) < n:
        while True:
            word = random.choice(list(no_sub_word.keys()))
            if filter(word):
                break
        sample_sw[word] = with_sub_word[word]
        sample_no_sw[word] = no_sub_word[word]
    return sample_sw, sample_no_sw

data1 = load_vectors("wiki-news-300d-1M.vec", 100)
data2 = load_vectors("wiki-news-300d-1M-subword.vec", 100)

sample1, sample2 = sample_words(data1, data2, 10, filter)

print(data1.keys())
print(len(sample1.keys()))
