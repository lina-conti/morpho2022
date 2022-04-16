#python
#morphology final project etc etc

import fasttext, editdistance, random
from fasttext import util
import numpy as np
import io
import re
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def cosine(veca, vecb):
    return (np.dot(veca, vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))

def squared_euclidian_distance(veca, vecb):
    res = 0
    for i in range(0, len(veca)):
        res = res + (veca[i] - vecb[i]) * (veca[i] - vecb[i])
    return res


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

def filter_catvar(fname):
    ''' creates a copy of the catvar file containing only the lines that contain derivational families and not a single word'''
    f_original = open(fname, 'r')
    f_new = open("derivational_families.txt", 'w')
    for line in f_original:
        words = line.split('#')
        if len(words) > 1:
            f_new.write(line)
    f_original.close()
    f_new.close()

def get_catvar_pairs(fname):
    ''' reads a catvar file and returns all pairs of morphologically related but not identical words it could find '''
    f = open(fname)
    for line in f:
        words = line.split('#')
        for i in range (len(words)):
            words[i] = words[i].split("_")[0]
        for i in range (len(words)):
    f.close()

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

def get_compare_pairs(sample1:dict, sample2:dict, num_comparisons):
    '''get the edit distance and cosine similarities in both datasets for a specified
    number of words to be compared'''
    words = []
    edit_distance = []
    cosine_dists = []

    for i in range(0, num_comparisons):
        w1 = random.sample(sample1.keys(), 1)[0] #random.sample returns a list of length 1, we take the item
        w2 = random.sample(sample2.keys(), 1)[0]
        if w1 != w2:
            words.append((w1, w2))
            edit_distance.append(editdistance.distance(w1, w2))
            cosine_dists.append(cosine(sample1[w1], sample2[w2]))
    edit_distance = np.array(edit_distance)[:, np.newaxis]

    return words, edit_distance, cosine_dists

def get_compare_pairs_ed(sample1:dict, sample2:dict, num_comparisons, return_nparray = True):
    '''get the edit distance and euclidian distances in both datasets for a specified
    number of words to be compared'''
    words = []
    edit_distance = []
    eucl_dists = []

    for i in range(0, num_comparisons):
        w1 = random.sample(sample1.keys(), 1)[0] #random.sample returns a list of length 1, we take the item
        w2 = random.sample(sample2.keys(), 1)[0]
        if w1 != w2:
            words.append((w1, w2))
            edit_distance.append(editdistance.distance(w1, w2))
            eucl_dists.append(squared_euclidian_distance(sample1[w1], sample2[w2]))
    if return_nparray: 
        edit_distance = np.array(edit_distance)[:, np.newaxis]

    return words, edit_distance, eucl_dists

def full_compare(sample: dict):
    '''runs a comprehensive comparison between every unique pair of words in a sample.
    returns the edit distances and cosine similarities'''
    words_a = sample.keys()
    words_b = sample.keys()
    e_dists = []
    cosine_dists = []
    for i, worda in enumerate(words_a):
        for wordb in list(words_b)[i:]:
            if wordb != worda:
                e_dists.append(editdistance.distance(worda, wordb))
                cosine_dists.append(cosine(sample[worda], sample[wordb]))
    return(e_dists, cosine_dists)
