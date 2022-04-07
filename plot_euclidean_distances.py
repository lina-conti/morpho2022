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
