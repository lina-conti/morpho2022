import functions
from functions import load_vectors, get_compare_pairs_ed
import numpy as np
import io
import re
import random
import scipy.stats as sp
from matplotlib import pyplot as plt

# ------------------------------- PLOT  -----------------------------------------------------

#load the samples
sw_sample_1 = load_vectors('morpho_project/morpho2022/sample_1_sw.vec', 1)
sw_sample_2 = load_vectors('morpho_project/morpho2022/sample_2_sw.vec', 1)

no_sw_sample_1 = load_vectors('morpho_project/morpho2022/sample_1_no_sw.vec', 1)
no_sw_sample_2 = load_vectors('morpho_project/morpho2022/sample_2_no_sw.vec', 1)

#comparisons for scipy
sp_sw_words, sp_sw_edists, sp_sw_eucl_d = get_compare_pairs_ed(sw_sample_1, sw_sample_2, 10000, return_nparray = False)
sp_no_sw_words, sp_no_sw_edists, sp_no_sw_eucl_d = get_compare_pairs_ed(no_sw_sample_1, no_sw_sample_2, 10000, return_nparray=False)

sp_with = sp.linregress(sp_sw_edists, sp_sw_eucl_d)
sp_without = sp.linregress(sp_no_sw_edists, sp_no_sw_eucl_d)


print('\nEUCLIDEAN DISTANCE')
print('\nwith subword')
print(f'slope: {sp_with.slope}\n\
r2 score: {sp_with.rvalue}\npval: {sp_with.pvalue}')
print('\nwithout subword')
print(f'slope: {sp_without.slope}\n\
r2 score: {sp_without.rvalue}\n\
pval: {sp_without.pvalue}')

"""
#plot actual values for each set
plt.scatter(sp_sw_edists, sp_sw_eucl_d, color = 'red')
plt.scatter(sp_no_sw_edists, sp_no_sw_eucl_d, color = 'blue')
"""

plt.plot(sp_sw_edists, sp_with.intercept + sp_with.slope*np.array(sp_sw_edists), 'r', label = 'with subwords')
plt.plot(sp_no_sw_edists, sp_without.intercept + sp_without.slope*np.array(sp_no_sw_edists),'b', label = 'without subwords')

"""
#get baselines: average euclidean distance for both models
plt.plot(sw_edists, [sum(sw_eucld)/len(sw_eucld)]*len(sw_edists),\
     color = 'pink', label = 'baseline with sw')
plt.plot(no_sw_edists, [sum(no_sw_eucld)/len(no_sw_eucld)]*len(no_sw_edists), \
    color = 'teal', label = 'baseline without sw')
"""

plt.title('Euclidean distance predicted by edit distance \n sample size of 2000 word, 10000 comparisons')
plt.xlabel('edit distance')
plt.ylabel('euclidean distance')
plt.legend()
plt.show()
