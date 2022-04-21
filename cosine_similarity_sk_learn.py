from functions import *

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

print('\nCOSINE SIMILARITY')
print('\nwith subword')
print(f'coefficient: {r_with.coef_}\n\
r2 score: {r_with.score(X_test_sw, y_test_sw)}')
print('\nwithout subword')
print(f'coefficient: {r_without.coef_}\n\
r2 score: {r_without.score(X_test_no_sw, y_test_no_sw)}\n')

"""
#plot regressor predictions for each dataset
plt.plot(sw_edists, r_with.predict(sw_edists), color = 'pink')
plt.plot(no_sw_edists, r_without.predict(no_sw_edists), color = 'teal')
"""
