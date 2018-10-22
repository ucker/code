# Dictionary Learning
from CoD import CoD
import numpy as np

def Wd(data, m, alpha = 0.5, threshold = 1e-4, iter_cnt = 600):
    '''

    Args:
        data :
        m : the sparse code dimension
        alpha : L1 normalization penalty coefficient
        threshold : as CoD function
        inter_cnt ：
    Returns:
        W : optimal dictionary
    '''
    N, n = data.shape
    W = np.random.randn(n, m)
    for i in range(m):
        # normalization column vector of dictionary matrix
        W[:, i] = W[:, i] / np.linalg.norm(W[:, i], 2)
    for k in range(iter_cnt):
        sample = data[k % N]
        # get optimal sparse code
        opti_Z = CoD(sample, W, alpha, threshold)
        # one step SGD
        W = W + 1 / (k + 1) * np.dot(np.array([sample - np.dot(W, opti_Z)]).T, np.array([opti_Z]))
        # normalization
        for i in range(m):
            W[:, i] = W[:, i] / np.linalg.norm(W[:, i], 2)
    return W