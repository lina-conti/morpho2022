# Advanced Morphology Project 2022

The goal of this project is to study [fasttext](https://fasttext.cc/) word embeddings with and without sub-word information.

We will be comparing pairs of words and trying to see how well their edit-distance correlates with the cosine similarity of their vector representations (with or without subword information). 

We then did the same thing using euclidean distance instead of cosine similarity.

We did all of this first using random pairs of words and then using only pairs of morphologically related words (taken from [catvar](https://github.com/nizarhabash1/catvar_).


## Python files: 
- functions.py contains the functions for reading and writing vector files and statistical analysis 
- files.py contains the code to read our sources, sample, filter and save relevant word_pairs and vectors

For active usage:
- cosine_similarity.py to compute the r2 values and plot the cosine similarities of unrelated words
- cosine_similarity_sk_learn.py: the same as the previous one but using sklearn instead of scipy 
- plot_euclidean_distances.py, r2_euclidean_distances.py: to use euclidean distances instead of cosine similarity
- morphologically_related.py computes all of the above but only for pairs of related words
- hist_cosine.py and his_euclidean_distances.py are used to show the distributions of these measures


