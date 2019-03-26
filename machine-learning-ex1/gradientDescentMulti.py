'''
GRADIENTDESCENTMULTI Performs gradient descent to learn theta

theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by taking num_iters gradient steps with learning rate alpha
'''
import numpy as np
from computeCost import computeCost


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = len(y)
    J_history = np.zeros((num_iters,1))

    for iter in range(num_iters):
        
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #         of the cost function (computeCostMulti) and gradient here.
        #
        
        theta = theta - (alpha/m) * np.matmul(np.transpose(X), (np.matmul(X, theta) - y))

        # ============================================================

         # Save the cost J in every iteration    

        J_history[iter] = computeCost(X, y, theta)    

    return theta, J_history