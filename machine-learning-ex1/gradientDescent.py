#Instructions: Perform a single gradient step on the parameter vector theta. 

import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters,1))

    for iter in range(num_iters):
        theta = theta - (alpha/m) * np.matmul(np.transpose(X), (np.matmul(X, theta) - y))

        J_history[iter] = computeCost(X, y, theta)    

    return theta