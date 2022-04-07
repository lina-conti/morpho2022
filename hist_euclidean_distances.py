import functions
from functions import load_vectors, get_compare_pairs_ed
import numpy as np
import io
import re
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# ------------------------------- HISTOGRAMS -----------------------------------------------------

#load the samples
sw_sample_1 = load_vectors('morpho_project/morpho2022/sample_1_sw.vec', 1)
sw_sample_2 = load_vectors('morpho_project/morpho2022/sample_2_sw.vec', 1)

no_sw_sample_1 = load_vectors('morpho_project/morpho2022/sample_1_no_sw.vec', 1)
no_sw_sample_2 = load_vectors('morpho_project/morpho2022/sample_2_no_sw.vec', 1)

#make comparisons
sw_words, sw_edists, sw_eucld = get_compare_pairs_ed(sw_sample_1, sw_sample_2, 10000)
no_sw_words, no_sw_edists, no_sw_eucld = get_compare_pairs_ed(no_sw_sample_1, no_sw_sample_2, 10000)


plt.hist(sw_eucld, bins = 50, label="with subword information") #results of random compare
plt.title('Full distribution of euclidean distances for 10000 pairs of vectors including subword information')
plt.xlabel('euclidean distance')
plt.ylabel('number of word pairs')
#plt.show()


plt.hist(no_sw_eucld, bins = 50, color = 'red', label= "without subword information") #random compare
plt.title('Full distribution of euclidean distances for 10,000 word pairs')
plt.xlabel('euclidean distance')
plt.ylabel('number of word pairs')
plt.legend()
plt.show()
