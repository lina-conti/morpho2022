import functions
from functions import load_vectors, get_compare_pairs_ed
import numpy as np
import io
import re
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import scipy.stats as sp


# ------------------------------- R^2 SCORES EUCLIDIAN DISTANCES -----------------------------------------------------

#load the samples
sw_sample_1 = load_vectors('sample_1_sw.vec', 1)
sw_sample_2 = load_vectors('sample_2_sw.vec', 1)

no_sw_sample_1 = load_vectors('sample_1_no_sw.vec', 1)
no_sw_sample_2 = load_vectors('sample_2_no_sw.vec', 1)

#make comparisons for sklearn
sw_words, sw_edists, sw_eucld = get_compare_pairs_ed(sw_sample_1, sw_sample_2, 10000)
no_sw_words, no_sw_edists, no_sw_eucld = get_compare_pairs_ed(no_sw_sample_1, no_sw_sample_2, 10000)

#comparisons for scipy
sp_sw_words, sp_sw_edists, sp_sw_eucld = get_compare_pairs_ed(sw_sample_1, sw_sample_2, 10000, return_nparray = False)
sp_no_sw_words, sp_no_sw_edists, sp_no_sw_eucld = get_compare_pairs_ed(no_sw_sample_1, no_sw_sample_2, 10000, return_nparray=False)

r_with = linear_model.LinearRegression()
r_without = linear_model.LinearRegression()


X_train_sw, X_test_sw, y_train_sw, y_test_sw = train_test_split(sw_edists, sw_eucld, test_size=0.2, random_state=42)
X_train_no_sw, X_test_no_sw, y_train_no_sw, y_test_no_sw = train_test_split(no_sw_edists, no_sw_eucld, test_size=0.2, random_state=42)

r_with.fit(X_train_sw, y_train_sw)
r_without.fit(X_train_no_sw, y_train_no_sw)

print(f"R^2 with subword: {r_with.score(X_test_sw, y_test_sw)}")
print(f"R^2 without subword: {r_with.score(X_test_no_sw, y_test_no_sw)}")


print(len(sw_edists), len(sw_eucld))
sp_with = sp.linregress(sp_sw_edists, sp_sw_eucld)
sp_without = sp.linregress(sp_no_sw_edists, sp_no_sw_eucld)

print(f'slope: {sp_with.slope}\n\
     r2 score for subword model: {sp_with.rvalue}\npval for subword model: {sp_with.pvalue}')

print(f'slope: {sp_without.slope}\n\
     r2 score for no subword model: {sp_without.rvalue}\n\
         pval for no subword model: {sp_without.pvalue}')
