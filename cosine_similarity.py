

from functions import *
import scipy.stats as sp
# ------------------------------- R^2 SCORES MAIN -----------------------------------------------------
''' Now using new random compare  '''

#load the samples
sw_sample_1 = load_vectors('sample_1_sw.vec', 1)
sw_sample_2 = load_vectors('sample_2_sw.vec', 1)

no_sw_sample_1 = load_vectors('sample_1_no_sw.vec', 1)
no_sw_sample_2 = load_vectors('sample_2_no_sw.vec', 1)

#make comparisons
sw_words, sw_edists, sw_cosines = get_compare_pairs(sw_sample_1, sw_sample_2, 10000)
no_sw_words, no_sw_edists, no_sw_cosines = get_compare_pairs(no_sw_sample_1, no_sw_sample_2, 10000)


r_with = linear_model.LinearRegression()
r_without = linear_model.LinearRegression()


X_train_sw, X_test_sw, y_train_sw, y_test_sw = train_test_split(sw_edists, sw_cosines, test_size=0.2, random_state=42)
X_train_no_sw, X_test_no_sw, y_train_no_sw, y_test_no_sw = train_test_split(no_sw_edists, no_sw_cosines, test_size=0.2, random_state=42)

r_with.fit(X_train_sw, y_train_sw)
r_without.fit(X_train_no_sw, y_train_no_sw)

#comparisons for scipy
sp_sw_words, sp_sw_edists, sp_sw_eucld = get_compare_pairs(sw_sample_1, sw_sample_2, 10000, return_nparray = False)
sp_no_sw_words, sp_no_sw_edists, sp_no_sw_eucld = get_compare_pairs(no_sw_sample_1, no_sw_sample_2, 10000, return_nparray=False)

sp_with = sp.linregress(sp_sw_edists, sp_sw_eucld)
sp_without = sp.linregress(sp_no_sw_edists, sp_no_sw_eucld)

print(f"R^2 with subword: {r_with.score(X_test_sw, y_test_sw)}")
print(f"R^2 without subword: {r_with.score(X_test_no_sw, y_test_no_sw)}")
"""
# ------------------------------- HISTOGRAMS -----------------------------------------------------
"""'''
plt.hist(sw_edists)
plt.title('Distribution of edit distances for 800 word pairs from a sample of 1000 words')
plt.xlabel('edit distance')
plt.ylabel('number of word pairs')
plt.show()

#plt.hist(cos_with, bins = 50) #results of full_compare
plt.hist(sw_cosines, bins = 50, label = 'with subword information') #results of random compare
plt.title('Full distribution of cosine similarities for a 500 word sample, subword information included')
plt.xlabel('cosine similarity')
plt.ylabel('number of word pairs')
#plt.show()

#plt.hist(cos_without, bins = 50) #full compare
plt.hist(no_sw_cosines, bins = 50, color = 'red', label = 'without subword information') #random compare
plt.title('Full distribution of subword information for a 500 word sample, no subword information')
plt.xlabel('cosine similarity')
plt.ylabel('number of word pairs')

plt.title('Full distribution of cosine similarities for 10,000 wordpairs')
plt.legend()
plt.show()'''
"""
# ------------------------------- PLOT MAIN -----------------------------------------------------

"""
#plot actual values for each set
#plt.scatter(sw_edists, sw_cosines, color = 'red', label = 'with subwords')
#plt.scatter(no_sw_edists, no_sw_cosines, color = 'blue', label = 'without subwords')

#plot regressor predictions for each dataset
plt.plot(sw_edists, r_with.predict(sw_edists), color = 'pink')
plt.plot(no_sw_edists, r_without.predict(no_sw_edists), color = 'teal')

print(f'slope: {sp_with.slope}\n\
     r2 score for subword model: {sp_with.rvalue}\npval for subword model: {sp_with.pvalue}')

print(f'slope: {sp_without.slope}\n\
     r2 score for no subword model: {sp_without.rvalue}\n\
         pval for no subword model: {sp_without.pvalue}')

plt.plot(sp_sw_edists, sp_with.intercept + sp_with.slope*np.array(sp_sw_edists), 'r', label = 'euclidean distance with subwords')
plt.plot(sp_no_sw_edists, sp_without.intercept + sp_without.slope*np.array(sp_no_sw_edists),'b', label = 'euclidean distance without subwords')

#get baselines: average cosine similarity for both models
'''plt.plot(sw_edists, [sum(sw_cosines)/len(sw_cosines)]*len(sw_edists),\
     color = 'pink', label = 'baseline with sw')
plt.plot(no_sw_edists, [sum(no_sw_cosines)/len(no_sw_cosines)]*len(no_sw_edists), \
    color = 'teal', label = 'baseline without sw')'''

plt.title('cosine similarity predicted by edit distance \n sample size of 2000 words, 10000 comparisons')
plt.xlabel('edit distance')
plt.ylabel('cosine similarity')
plt.legend()
plt.show()

