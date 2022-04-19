import functions
from functions import load_vectors, get_compare_pairs_ed, filter_catvar, get_catvar_pairs, write_word_pairs, read_word_pairs, words_to_features
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# filter_catvar("catvar21.signed")
# pairs = get_catvar_pairs("derivational_families.txt")
# write_word_pairs("related_pairs.txt", pairs)

'''
word_pairs, edit_d, cosine_s_sw, euclid_d_sw = words_to_features(TODO, TODO)
word_pairs, edit_d, cosine_s_no_sw, euclid_d_no_sw = words_to_features(TODO, TODO)
'''
