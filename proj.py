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

def get_compare_pairs(sample1:dict, sample2:dict, num_comparisons):
    '''get the edit distance and cosine similarities in both datasets for a specified
    number of words to be compared'''
    words = []
    edit_distance = []
    s1_cosine_dists = []
    s2_cosine_dists = []

    for i in range(0, num_comparisons):
        w1 = random.sample(sample1.keys(), 1)[0] #random.sample returns a list of length 1, we take the item
        w2 = random.sample(sample2.keys(), 1)[0]
        if w1 != w2:
            words.append((w1, w2))
            edit_distance.append(editdistance.distance(w1, w2))
            s1_cosine_dists.append(cosine(sample1[w1], sample1[w2]))
            s2_cosine_dists.append(cosine(sample2[w1], sample2[w2]))
    edit_distance = np.array(edit_distance)[:, np.newaxis]

    return words, edit_distance, s1_cosine_dists, s2_cosine_dists

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

# ------------------------------- WRITING OUR SAMPLES TO A FILE -----------------------------------------------------
'''
top_vectors_sw = load_vectors("wiki-news-300d-1M-subword.vec", 0.1)
sample_1_sw, sample_1_no_sw = sample_words(top_vectors_sw, top_vectors_no_sw, 1000, filter)
sample_2_sw, sample_2_no_sw = sample_words(top_vectors_sw, top_vectors_no_sw, 1000, filter)

# writes the samples to files to keep them for future use
write_to_file("morpho2022/sample_1_sw.vec", sample_1_sw)
write_to_file("morpho2022/sample_1_no_sw.vec", sample_1_no_sw)
write_to_file("morpho2022/sample_2_sw.vec", sample_2_sw)
write_to_file("morpho2022/sample_2_no_sw.vec", sample_2_no_sw)
'''

# ------------------------------- R^2 SCORES MAIN -----------------------------------------------------
''' Now using new random compare  '''

'''
#load the samples
sw_sample_1 = load_vectors('morpho_project/morpho2022/sample_1_sw.vec', 1)
sw_sample_2 = load_vectors('morpho_project/morpho2022/sample_2_sw.vec', 1)

no_sw_sample_1 = load_vectors('morpho_project/morpho2022/sample_1_no_sw.vec', 1)
no_sw_sample_2 = load_vectors('morpho_project/morpho2022/sample_2_no_sw.vec', 1)

#make comparisons
sw_words, sw_edists, sw_cosines = get_compare_pairs(sw_sample_1, sw_sample_2, 10000)
no_sw_words, no_sw_edists, no_sw_cosines = get_compare_pairs(no_sw_sample_1, no_sw_sample_2, 10000)


r_with = linear_model.LinearRegression()
r_without = linear_model.LinearRegression()


X_train_sw, X_test_sw, y_train_sw, y_test_sw = train_test_split(sw_edists, sw_cosines, test_size=0.2, random_state=42)
X_train_no_sw, X_test_no_sw, y_train_no_sw, y_test_no_sw = train_test_split(no_sw_edists, no_sw_cosines, test_size=0.2, random_state=42)

r_with.fit(X_train_sw, y_train_sw)
r_without.fit(X_train_no_sw, y_train_no_sw)

print(f"R^2 with subword: {r_with.score(X_test_sw, y_test_sw)}")
print(f"R^2 without subword: {r_with.score(X_test_no_sw, y_test_no_sw)}")


# ------------------------------- HISTOGRAMS -----------------------------------------------------

plt.hist(sw_edists)
plt.title('Distribution of edit distances for 800 word pairs from a sample of 1000 words')
plt.xlabel('edit distance')
plt.ylabel('number of word pairs')
plt.show()

#plt.hist(cos_with, bins = 50) #results of full_compare
plt.hist(sw_cosines, bins = 50) #results of random compare
plt.title('Full distribution of cosine similarities for a 500 word sample, subword information included')
plt.xlabel('cosine similarity')
plt.ylabel('number of word pairs')
#plt.show()

#plt.hist(cos_without, bins = 50) #full compare
plt.hist(no_sw_cosines, bins = 50, color = 'red') #random compare
plt.title('Full distribution of subword information for a 500 word sample, no subword information')
plt.xlabel('cosine similarity')
plt.ylabel('number of word pairs')
plt.show()

# ------------------------------- PLOT MAIN -----------------------------------------------------


#plot actual values for each set
plt.scatter(sw_edists, sw_cosines, color = 'red', label = 'with subwords')
plt.scatter(no_sw_edists, no_sw_cosines, color = 'blue', label = 'without subwords')

#plot regressor predictions for each dataset
plt.plot(sw_edists, r_with.predict(sw_edists), color = 'red')
plt.plot(no_sw_edists, r_without.predict(no_sw_edists), color = 'blue')

#get baselines: average cosine similarity for both models
plt.plot(sw_edists, [sum(sw_cosines)/len(sw_cosines)]*len(sw_edists),\
     color = 'pink', label = 'baseline with sw')
plt.plot(no_sw_edists, [sum(no_sw_cosines)/len(no_sw_cosines)]*len(no_sw_edists), \
    color = 'teal', label = 'baseline without sw')

plt.title('cosine similarity predicted by edit distance \n sample size of 1000 word, 1000 comparisons')
plt.xlabel('edit distance')
plt.ylabel('cosine similarity')
plt.legend()
plt.show()
'''

