from functions import *
import scipy.stats as sp



word_pairs, edit_d, cosine_s_sw, euclid_d_sw = words_to_features("morpho_project/morpho2022/pairs_sw.vec", "morpho_project/morpho2022/sampled_pairs")
word_pairs, edit_d, cosine_s_no_sw, euclid_d_no_sw = words_to_features("morpho_project/morpho2022/pairs_no_sw.vec", "morpho_project/morpho2022/sampled_pairs")

sp_cosine_with = sp.linregress(edit_d[:10000], cosine_s_sw[:10000])
sp_cosine_without = sp.linregress(edit_d[:10000], cosine_s_no_sw[:10000])

sp_euclid_with = sp.linregress(edit_d[:10000], euclid_d_sw[:10000])
sp_euclid_without = sp.linregress(edit_d[:10000], euclid_d_no_sw[:10000])

# ------------------------------- SLOPE, R2 AND PVALUE  -----------------------------------------------------

print('\nCOSINE SIMILARITY')
print('\nwith subword')
print(f'slope: {sp_cosine_with.slope}\n\
r2 score: {sp_cosine_with.rvalue}\npval: {sp_cosine_with.pvalue}')
print('\nwithout subword')
print(f'slope: {sp_cosine_without.slope}\n\
r2 score: {sp_cosine_without.rvalue}\n\
pval: {sp_cosine_without.pvalue}')

print('\n\nEUCLIDIAN DISTANCE')
print('\nwith subword')
print(f'slope: {sp_euclid_with.slope}\n\
r2 score: {sp_euclid_with.rvalue}\npval: {sp_euclid_with.pvalue}')
print('\nwithout subword')
print(f'slope: {sp_euclid_without.slope}\n\
r2 score: {sp_euclid_without.rvalue}\n\
pval: {sp_euclid_without.pvalue}')

# ------------------------------- PLOTTING COSINE  -----------------------------------------------------

plt.plot(edit_d[:10000], sp_cosine_with.intercept + sp_cosine_with.slope*np.array(edit_d[:10000]), 'r', label = 'with subwords')
plt.plot(edit_d[:10000], sp_cosine_without.intercept + sp_cosine_without.slope*np.array(edit_d[:10000]),'b', label = 'without subwords')

#plot actual values for each set
plt.scatter(edit_d[:10000], cosine_s_sw[:10000], color = 'red')
plt.scatter(edit_d[:10000], cosine_s_no_sw[:10000], color = 'blue')

plt.title('Cosine similarity between the vector representation of pairs of morphologically related words predicted from their edit distance')
plt.xlabel('edit distance')
plt.ylabel('cosine similarity')
plt.legend()
plt.show()

# ------------------------------- PLOTTING EUCLIDEAN  -----------------------------------------------------

plt.plot(edit_d[:10000], sp_euclid_with.intercept + sp_euclid_with.slope*np.array(edit_d[:10000]), 'r', label = 'with subwords')
plt.plot(edit_d[:10000], sp_euclid_without.intercept + sp_euclid_without.slope*np.array(edit_d[:10000]),'b', label = 'without subwords')

#plot actual values for each set
plt.scatter(edit_d[:10000], euclid_d_sw[:10000], color = 'red')
plt.scatter(edit_d[:10000], euclid_d_no_sw[:10000], color = 'blue')

plt.title('Euclidean distance between the vector representation of pairs of morphologically related words predicted from their edit distance')
plt.xlabel('edit distance')
plt.ylabel('euclidean distance')
plt.legend()
plt.show()
