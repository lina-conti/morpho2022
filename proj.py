#python 
#morphology final project etc etc 


import fasttext, editdistance
from fasttext import util
import numpy as np 


print(editdistance.distance('cat', 'bat'))

def cosine(veca, vecb):
    return (np.dot(veca, vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))

print(cosine([1, 2, 1], [1, 1, 1]))


