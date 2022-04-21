from functions import *

#load the samples
sw_sample_1 = load_vectors('sample_1_sw.vec', 1)
sw_sample_2 = load_vectors('sample_2_sw.vec', 1)

no_sw_sample_1 = load_vectors('sample_1_no_sw.vec', 1)
no_sw_sample_2 = load_vectors('sample_2_no_sw.vec', 1)

#make comparisons
sw_words, sw_edists, sw_cosines = get_compare_pairs(sw_sample_1, sw_sample_2, 10000)
no_sw_words, no_sw_edists, no_sw_cosines = get_compare_pairs(no_sw_sample_1, no_sw_sample_2, 10000)

plt.hist(sw_edists)
plt.title('Distribution of edit distances for 10000 word pairs from a sample of 2000 words')
plt.xlabel('edit distance')
plt.ylabel('number of word pairs')
plt.show()


plt.hist(sw_cosines, bins = 50, label = 'with subword information') #results of random compare
plt.title('Full distribution of cosine similarities for 10000 word pairs, subword information included')
plt.xlabel('cosine similarity')
plt.ylabel('number of word pairs')
#plt.show()


plt.hist(no_sw_cosines, bins = 50, color = 'red', label = 'without subword information') #random compare
plt.title('Full distribution of cosine similarities for 10000 word pairs, no subword information')
plt.xlabel('cosine similarity')
plt.ylabel('number of word pairs')

plt.title('Full distribution of cosine similarities for 10,000 wordpairs')
plt.legend()
plt.show()
