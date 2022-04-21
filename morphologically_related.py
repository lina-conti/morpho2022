import functions
from functions import *
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt




word_pairs, edit_d, cosine_s_sw, euclid_d_sw = words_to_features("morpho_project/morpho2022/pairs_sw.vec", "morpho_project/morpho2022/sampled_pairs")
word_pairs, edit_d, cosine_s_no_sw, euclid_d_no_sw = words_to_features("morpho_project/morpho2022/pairs_no_sw.vec", "morpho_project/morpho2022/sampled_pairs")

print(word_pairs)
print(len(edit_d))
print(len(cosine_s_sw))
print(len(cosine_s_no_sw))
print(len(euclid_d_sw))
print(len(euclid_d_no_sw))
