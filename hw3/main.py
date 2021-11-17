from scipy.sparse.csr import csr_matrix
import config
from scipy import sparse
import numpy as np

def make_prediction(X, w, V):
    a = np.sum(np.square(X.dot(V)), axis=1).reshape(-1,1)
    b = np.sum(X.power(2).dot(np.square(V)), axis=1).reshape(-1,1)
    return X.dot(w) + 0.5 * (a - b)

S: csr_matrix = sparse.load_npz('{0}.npz'.format(config.csr_path))
print(S.shape)
_, features = S.shape
V = np.full((features, config.k), 0.5)
print(V.shape)
print(S.dot(V))
