
# ------------------------------- WRITING OUR SAMPLES TO A FILE -----------------------------------------------------
"""
top_vectors_sw = load_vectors("wiki-news-300d-1M-subword.vec", 0.1)
top_vectors_no_sw = load_vectors("wiki-news-300d-1M.vec", 0.1)
sample_1_sw, sample_1_no_sw = sample_words(top_vectors_sw, top_vectors_no_sw, 1000, filter)
sample_2_sw, sample_2_no_sw = sample_words(top_vectors_sw, top_vectors_no_sw, 1000, filter)

# writes the samples to files to keep them for future use
write_to_file("morpho_project/morpho2022/sample_1_sw.vec", sample_1_sw)
write_to_file("morpho_project/morpho2022/sample_1_no_sw.vec", sample_1_no_sw)
write_to_file("morpho_project/morpho2022/sample_2_sw.vec", sample_2_sw)
write_to_file("morpho_project/morpho2022/sample_2_no_sw.vec", sample_2_no_sw)
"""

# ------------------------------- R^2 SCORES EUCLIDIAN DISTANCES -----------------------------------------------------
''' Now using new random compare  '''

#load the samples
sw_sample_1 = load_vectors('morpho_project/morpho2022/sample_1_sw.vec', 1)
sw_sample_2 = load_vectors('morpho_project/morpho2022/sample_2_sw.vec', 1)

no_sw_sample_1 = load_vectors('morpho_project/morpho2022/sample_1_no_sw.vec', 1)
no_sw_sample_2 = load_vectors('morpho_project/morpho2022/sample_2_no_sw.vec', 1)

#make comparisons
sw_words, sw_edists, sw_eucld = get_compare_pairs_ed(sw_sample_1, sw_sample_2, 10000)
no_sw_words, no_sw_edists, no_sw_eucld = get_compare_pairs_ed(no_sw_sample_1, no_sw_sample_2, 10000)


r_with = linear_model.LinearRegression()
r_without = linear_model.LinearRegression()


X_train_sw, X_test_sw, y_train_sw, y_test_sw = train_test_split(sw_edists, sw_eucld, test_size=0.2, random_state=42)
X_train_no_sw, X_test_no_sw, y_train_no_sw, y_test_no_sw = train_test_split(no_sw_edists, no_sw_eucld, test_size=0.2, random_state=42)

r_with.fit(X_train_sw, y_train_sw)
r_without.fit(X_train_no_sw, y_train_no_sw)

print(f"R^2 with subword: {r_with.score(X_test_sw, y_test_sw)}")
print(f"R^2 without subword: {r_with.score(X_test_no_sw, y_test_no_sw)}")

# ------------------------------- R^2 SCORES MAIN -----------------------------------------------------
''' Now using new random compare  '''
"""
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
"""
# ------------------------------- HISTOGRAMS -----------------------------------------------------
"""
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
"""
# ------------------------------- PLOT MAIN -----------------------------------------------------

"""
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
"""
