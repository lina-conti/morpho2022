import functions
from functions import load_vectors, get_compare_pairs_ed, filter_catvar, get_catvar_pairs, write_word_pairs
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#filter_catvar("catvar21.signed")
pairs = get_catvar_pairs("derivational_families.txt")
print(pairs)
print(len(pairs))
write_word_pairs("related_pairs.txt", pairs)
