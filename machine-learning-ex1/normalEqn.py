'''
NORMALEQN Computes the closed-form solution to linear regression 
NORMALEQN(X,y) computes the closed-form solution to linear regression using the normal equations.
'''
import numpy as np
def normalEqn(X,y):

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    X_transpose = np.transpose(X)
    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_transpose, X)), X_transpose),y)

# ============================================================
    return theta
