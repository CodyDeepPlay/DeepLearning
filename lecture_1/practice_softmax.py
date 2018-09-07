# -*- coding: utf-8 -*-
"""Softmax."""

scores = [3.0, 1.0, 0.2]  # score is a litst object

import numpy as np
#%%
'''
Turn the scores into the probability.

refer to this website for a quick view of the function:
    https://classroom.udacity.com/courses/ud730/lessons/6370362152/concepts/63798118200923


'''

def softmax(scores): 
    exp_scores = np.exp(scores)  # calculate exponentional of all the scores
    sum_exp_scores = np.sum(exp_scores, axis=0)
    probability_all= exp_scores/sum_exp_scores    
    return probability_all
    
#np.exp(scores)/np.sum(np.exp(scores), axis=0)


"""Compute softmax values for each sets of scores in x."""
#   exp_scores   = [];  # create a empty list
#   final_scores = []
#   sum_scores   = 0; 
#   for i in range(len(scores)): 
#       exp_scores.append ( np.exp(scores[i]) );     # math.exp is the operation for exponential with natural base, "**" is the operation for exponential 
#       sum_scores = sum_scores + exp_scores[i];       # compute the denomenater for the softmax equation
#
#   for j in range(len(scores)): 
#       final_scores.append( exp_scores[j] / sum_scores ); # compute the score of each related input
#              
#   #pass final_scores # TODO: Compute and return softmax(x)
#   return final_scores
   
   
"""or use this one sentence, could do the same job"""
   #return np.exp(scores)/np.sum(np.exp(scores),axis = 0);
   

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)                                        # x varies from -2 to 6, stepwise is 0.1
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])      # scores is a 3 by 80 numpy.narray

plt.plot(x, softmax(scores).T, linewidth=2)                          # .Ｔ is transposing the matrix
plt.show()


