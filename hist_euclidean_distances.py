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
