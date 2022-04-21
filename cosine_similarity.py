from functions import *
import scipy.stats as sp


#load the samples
sw_sample_1 = load_vectors('sample_1_sw.vec', 1)
sw_sample_2 = load_vectors('sample_2_sw.vec', 1)

no_sw_sample_1 = load_vectors('sample_1_no_sw.vec', 1)
no_sw_sample_2 = load_vectors('sample_2_no_sw.vec', 1)

#comparisons for scipy
sp_sw_words, sp_sw_edists, sp_sw_cos = get_compare_pairs(sw_sample_1, sw_sample_2, 10000, return_nparray = False)
sp_no_sw_words, sp_no_sw_edists, sp_no_sw_cos = get_compare_pairs(no_sw_sample_1, no_sw_sample_2, 10000, return_nparray=False)

sp_with = sp.linregress(sp_sw_edists, sp_sw_cos)
sp_without = sp.linregress(sp_no_sw_edists, sp_no_sw_cos)


print('\nCOSINE SIMILARITY')
print('\nwith subword')
print(f'slope: {sp_with.slope}\n\
r2 score: {sp_with.rvalue}\npval: {sp_with.pvalue}')
print('\nwithout subword')
print(f'slope: {sp_without.slope}\n\
r2 score: {sp_without.rvalue}\n\
pval: {sp_without.pvalue}')


"""#plot actual values for each set
plt.scatter(sp_sw_edists, sp_sw_cos, color = 'red')
plt.scatter(sp_no_sw_edists, sp_no_sw_cos, color = 'blue')
"""
plt.plot(sp_sw_edists, sp_with.intercept + sp_with.slope*np.array(sp_sw_edists), 'r', label = 'with subwords')
plt.plot(sp_no_sw_edists, sp_without.intercept + sp_without.slope*np.array(sp_no_sw_edists),'b', label = 'without subwords')

#get baselines: average cosine similarity for both models
'''
plt.plot(sw_edists, [sum(sw_cosines)/len(sw_cosines)]*len(sw_edists),\
     color = 'pink', label = 'baseline with sw')
plt.plot(no_sw_edists, [sum(no_sw_cosines)/len(no_sw_cosines)]*len(no_sw_edists), \
    color = 'teal', label = 'baseline without sw')
'''

plt.title('Cosine similarity predicted from edit distance \n(10000 word pairs, from a sample 2000 words)')
plt.xlabel('edit distance')
plt.ylabel('cosine similarity')
plt.legend()
plt.show()
