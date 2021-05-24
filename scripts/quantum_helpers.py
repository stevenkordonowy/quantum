import numpy as np
from functools import reduce
from math import sqrt

I = np.array([[1,0],[0,1]], int)
Z = np.array([[1,0],[0,-1]], int)
X = np.array([[0,1],[1,0]], int)
plus_ket = np.array([[1],[1]])/sqrt(2)

def M_K(K, n, M):
    individual_operators = [M if (gate in K) else I for gate in range(n)]
    return reduce(np.kron, individual_operators)

def plus_n(n):
    return M_K(range(n), n, plus_ket)
