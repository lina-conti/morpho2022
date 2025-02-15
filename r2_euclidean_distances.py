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

#load the samples
sw_sample_1 = load_vectors('sample_1_sw.vec', 1)
sw_sample_2 = load_vectors('sample_2_sw.vec', 1)

no_sw_sample_1 = load_vectors('sample_1_no_sw.vec', 1)
no_sw_sample_2 = load_vectors('sample_2_no_sw.vec', 1)

# ------------------------------- WITH SKLEARN -----------------------------------------------------

#make comparisons for sklearn
sw_words, sw_edists, sw_eucld = get_compare_pairs_ed(sw_sample_1, sw_sample_2, 10000)
no_sw_words, no_sw_edists, no_sw_eucld = get_compare_pairs_ed(no_sw_sample_1, no_sw_sample_2, 10000)

r_with = linear_model.LinearRegression()
r_without = linear_model.LinearRegression()

X_train_sw, X_test_sw, y_train_sw, y_test_sw = train_test_split(sw_edists, sw_eucld, test_size=0.2, random_state=42)
X_train_no_sw, X_test_no_sw, y_train_no_sw, y_test_no_sw = train_test_split(no_sw_edists, no_sw_eucld, test_size=0.2, random_state=42)

r_with.fit(X_train_sw, y_train_sw)
r_without.fit(X_train_no_sw, y_train_no_sw)

print("SKLEARN\n")
print(f"R^2 with subword: {r_with.score(X_test_sw, y_test_sw)}")
print(f"R^2 without subword: {r_without.score(X_test_no_sw, y_test_no_sw)}")

# ------------------------------- WITH SCIPY -----------------------------------------------------

#comparisons for scipy
sp_sw_words, sp_sw_edists, sp_sw_eucld = get_compare_pairs_ed(sw_sample_1, sw_sample_2, 10000, return_nparray = False)
sp_no_sw_words, sp_no_sw_edists, sp_no_sw_eucld = get_compare_pairs_ed(no_sw_sample_1, no_sw_sample_2, 10000, return_nparray=False)

sp_with = sp.linregress(sp_sw_edists, sp_sw_eucld)
sp_without = sp.linregress(sp_no_sw_edists, sp_no_sw_eucld)

print("\nSCIPY")
print('\nwith subword')
print(f'slope: {sp_with.slope}\n\
r2 score: {sp_with.rvalue}\
\npval: {sp_with.pvalue}')
print('\nwithout subword')
print(f'slope: {sp_without.slope}\n\
r2 score: {sp_without.rvalue}\n\
pval: {sp_without.pvalue}')

# ------------------------------- COMPARISON PLOT -----------------------------------------------------

plt.plot(sp_sw_edists, sp_with.intercept + sp_with.slope*np.array(sp_sw_edists), 'r', label = 'euclidean distance with subwords')
plt.plot(sp_no_sw_edists, sp_without.intercept + sp_without.slope*np.array(sp_no_sw_edists),'b', label = 'euclidean distance without subwords')
plt.plot(sw_edists, r_with.predict(sw_edists), color = 'pink', label = 'sklearn with subword informtation')
plt.plot(no_sw_edists, r_without.predict(no_sw_edists), color = 'teal', label = 'sklearn without subword information')
plt.legend()
plt.show()
