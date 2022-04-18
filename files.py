from functions import *

# ------------------------------- WRITING OUR SAMPLES TO A FILE -----------------------------------------------------

'''top_vectors_sw = load_vectors("wiki-news-300d-1M-subword.vec", 0.1)
top_vectors_no_sw = load_vectors("wiki-news-300d-1M.vec", 0.1)
sample_1_sw, sample_1_no_sw = sample_words(top_vectors_sw, top_vectors_no_sw, 1000, filter)
sample_2_sw, sample_2_no_sw = sample_words(top_vectors_sw, top_vectors_no_sw, 1000, filter)

# writes the samples to files to keep them for future use
write_to_file("morpho_project/morpho2022/sample_1_sw.vec", sample_1_sw)
write_to_file("morpho_project/morpho2022/sample_1_no_sw.vec", sample_1_no_sw)
write_to_file("morpho_project/morpho2022/sample_2_sw.vec", sample_2_sw)
write_to_file("morpho_project/morpho2022/sample_2_no_sw.vec", sample_2_no_sw)
'''
#test_words = read_word_pairs('related_pairs.txt')
test_with, test_without, test_words = sample_wordpairs('related_pairs.txt', 'wiki-news-300d-1M-subword.vec', 
'wiki-news-300d-1M.vec', prop = 0.1)

write_to_file('pairs_sw.vec', test_with)
write_to_file('pairs_no_sw.vec', test_without)
write_word_pairs('sampled_pairs', test_words)

print(test_words[:3])

