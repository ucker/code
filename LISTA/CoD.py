# Coordinate Descent Algorithm
import numpy as np 

def softshrink(lnput, lambd=0.5):
    #print(lnput, lambd)
    if abs(lnput) - lambd> 0:
        return lnput / abs(lnput) * (abs(lnput) - lambd)
    else:
        return 0

def vector_softshrink(lnput, lambd=0.5):
    res = []
    # print(lnput)
    for i in range(len(lnput)):
        res.append(softshrink(lnput[i], lambd))
    return np.array(res)

def CoD(X, W, alpha, threshold=1e-5, h=vector_softshrink):
    '''
    Args:
        X : data (size n vector)
        W : dictionary matrix (n * m matrix) column vector is unit
        alpha : L1 normalization penalty coefficient
        threshold : threshold of Z
        h : shrinkage function
    Returns:
        Z : sparse code of X
    '''
    n, m = W.shape
    B = np.dot(W.T, X)
    S = np.eye(m) - np.dot(W.T, W)
    Z = np.zeros(m)
    dz = np.inf * np.ones(m)
    while True:
        Z_1 = h(B, alpha)
        dz = Z_1 - Z
        k = np.argmax(np.abs(dz))
        B = B + dz[k] * S[:, k]
        Z[k] = Z_1[k]
        if np.abs(dz[k]) < threshold:
            break
    Z = h(B, alpha)
    return Z
    