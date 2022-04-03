#python
#morphology final project etc etc


import fasttext, editdistance, random
from fasttext import util
import numpy as np
import io
import re
import random

def cosine(veca, vecb):
    return (np.dot(veca, vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))



# creates a dictionary ([str -> map object]) that maps a word to its vector representation
# by reading fname file (which should be in the .vec format of fasttext)
# proportion indicates the proportion of word vectors we want to read from the file (a float between 0 and 1)
def load_vectors(fname, proportion):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split()) # n is the number of vectors, d is the vector size
    data = {}
    i = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
        i += 1
        if i >= n * proportion:
            break
    fin.close()
    return data

def filter(word):
    punct = re.compile('[^a-z]')
    if punct.search(word):
        return False
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
                #print('illegal symbol detected')
                break
        sample_sw[word] = with_sub_word[word]
        sample_no_sw[word] = no_sub_word[word]
    return sample_sw, sample_no_sw

# sample is a dictionary from words to vectors
# it will be written to a .vec file following fasttext conventions
# content already present in fname will be overwritten, fname will be created if it doesn't exist
def write_to_file(fname, sample):
    f = open(fname, "w")
    f.write(f"{len(sample)} 300")
    for word in sample:
        f.write(f"\n{word}")
        for component in sample[word]:
            f.write(f" {component}")
    f.close()

def get_compare_pairs(sample1:dict, sample2:dict):
    words = []
    edit_distance = []
    s1_cosine_dists = []
    s2_cosine_dists = []

    w1 = random.sample(sample1.keys(), 1)[0] #random.sample returns a list of length 1, we take the item
    w2 = random.sample(sample1.keys(), 1)[0]

    #print(f'word: {w1}, vec: {sample1[w1]}')
    words.append((w1, w2))
    edit_distance.append(editdistance.distance(w1, w2))
    s1_cosine_dists.append(cosine(sample1[w1], sample2[w2]))
    s2_cosine_dists.append(cosine(sample2[w1], sample2[w2]))

    return words, edit_distance, s1_cosine_dists, s2_cosine_dists

# ------------------------------- LINA'S MAIN -----------------------------------------------------

data_set_no_sw = load_vectors("wiki-news-300d-1M.vec", 0.5)
data_set_sw = load_vectors("wiki-news-300d-1M-subword.vec", 0.5)

sample_sw, sample_no_sw = sample_words(data1, data2, 10, filter)

write_to_file("sample_sw.vec", sample_sw)
write_to_file("sample_no_sw.vec", sample_no_sw)

# ------------------------------- ISAAC'S MAIN -----------------------------------------------------


data1 = load_vectors("wiki-news-300d-1M.vec", 0.01)
data2 = load_vectors("wiki-news-300d-1M-subword.vec", 0.01)

sample1, sample2 = sample_words(data1, data2, 10, filter)

print(sample1.keys())
print(len(sample1.keys()))

words, e_dists, s1_cosines, s2_cosines = get_compare_pairs(sample1, sample2)
print(words, e_dists, s1_cosines, s2_cosines)
#python
#morphology final project etc etc


write_to_file("test.vec", sample1)

test = load_vectors("test.vec", 1)
for i,j in test.items():
    print(i)
print("\n\n\n")
for i in sample1.items():
    print(i)
