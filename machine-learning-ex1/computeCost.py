'''
COMPUTECOST Compute cost for linear regression
J = COMPUTECOST(X, y, theta) computes the cost of using theta as the parameter for linear regression to fit the data points in X and y
'''
import numpy as np


def computeCost(X, Y, theta):
    # # Initialize some useful values
    m = len(Y) #number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    J = np.sum( np.square(np.matmul(X, theta)-Y))/ (2* m)
   

    # =========================================================================
    return J