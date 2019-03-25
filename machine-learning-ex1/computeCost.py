'''
Instructions: Compute the cost of a particular choice of theta
              You should set J to the cost.
'''
import numpy as np
def computeCost(X, Y, theta):
    m = len(Y) #number of training examples
    J = np.sum( np.square(np.matmul(X, theta)-Y))/ (2* m)
   
    return J