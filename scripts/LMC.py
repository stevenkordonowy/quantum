import numpy as np
# import gnumpy as gpu
import math, cmath
from scipy.linalg import expm
from functools import reduce
# import itertools
import time
from numpy.linalg import norm
import ring
from qutip import *

I = np.array([[1,0],[0,1]], int)
X = np.array([[0,1],[1,0]], int)
plus = np.array([[1],[1]], int) / np.sqrt(2)
H2_5 = ring.construct_ham_local2_as_arr(2, 5)
# P_perm = np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,0]],int).T

'''
Constructs the uniform superposition of n qubits
'''
def plus_n(n):
    return reduce(np.kron, [plus for _ in range(n)])

'''
Constructs the operator that is the X NOT on the i'th qubit
and the identity on the others
'''
def Xi(i, n):
    individual_operators = [X if gate == i else I for gate in range(n)]
    return reduce(np.kron, individual_operators)

'''
e^(i * angle * H)
'''
def expi(U, angle):
    return expm(-1j * angle * U)

'''
Calculates operator sum(Xi), where Xi is the NOT operator on the i'th bit and the identity on the remaining
'''
def X_sum(n):
    return reduce(np.add, [Xi(i,n) for i in range(n)])

def hamming_dist(n1, n2) : 

    x = n1 ^ n2  
    setBits = 0

    while (x > 0) : 
        d = x & 1
        setBits += d
        x >>= 1
    
    return setBits  



def H_jj_ring_fast(cut, n):
    if n == 5:
        return H_jj_ring_fast_5(cut)
    
    cut_in_5 = ring.project(cut, n)
    num_happy = H2_5[cut_in_5]

    for k in range(1, n):
        cut_in_5 = ring.shift(cut, k, n)
        rotated_j_in_5 = ring.project(cut_in_5, n)
        num_happy = num_happy + H2_5[rotated_j_in_5]
    return num_happy / 3


def H_jj_ring_fast_5(cut):
    num_happy = H2_5[cut]

    for k in range(1, 5):
        cut_in_5 = ring.shift(cut, k, 5)
        num_happy = num_happy + H2_5[cut_in_5]
    return num_happy / 3



'''
Returns U = Ub_n*Uc_n* ... *Ub_1*Uc_1
'''
def construct_unitary(U_gammas, U_betas):
    Us = []
    for i in range(len(U_gammas)):
        # Gotta be a smarter way, zip?
        Us.append(U_betas[i])
        Us.append(U_gammas[i])

    U = reduce(np.dot, Us)
    return U


def construct_U_beta(beta, n):
    Ub = expm(-1j*beta*X)
    return reduce(np.kron, [Ub for _ in range(n)])

def construct_U_gamma(gamma, n):
    d = [cmath.exp(-1j*gamma*H_jj_ring_fast(k, n)) for k in range(2 ** n)]
    return np.diag(d)

def construct_U_beta_qutip(beta, n):
    Ub = Qobj(expm(-1j*beta*X), copy=False)
    return tensor([Ub for _ in range(n)])

def construct_U_gamma_qutip(gamma, n):
    d = [cmath.exp(-1j*gamma*H_jj_ring_fast(k, n)) for k in range(2 ** n)]
    return Qobj(np.diag(d), copy=False)


def U_angle_operator(U, angle):
    return expi(U, angle)


''' 
Calcuates expectations of the local max cut problem using the QAOA algorithm 
'''    
class LocalMaxCut:

    def __init__(self, H):
        self.Ham = H

        self.__N = len(H[0])
        self.__n = int(math.log2(self.__N))
        self.X_sum = X_sum(self.__n)
        self.plus_n = plus_n(self.__n)
        self.__ExpHam = np.array(np.empty(2 ** 5), dtype='complex')
        self.__ExpHam[:] = np.nan

    # def expected_value_p(self, gammas, betas):

    def U_gamma_operator(self, gamma):
        return U_angle_operator(self.Ham, gamma)

    def U_beta_operator(self, beta):
        return U_angle_operator(self.X_sum, beta)

        
    '''
    Calculates <+|U.dag * H * U|+>
    '''
    def schrodinger_ev(self, U_gammas, U_betas):
        U = construct_unitary(U_gammas, U_betas)
        psi = U.dot(self.plus_n)
        exp = psi.conj().T.dot(self.Ham).dot(psi)
        return abs(exp.item())
    


    # def U_gamma_eigenval(self, ):

    def U_beta_entry(self, beta, i, j):
        beta_sin = math.sin(beta)
        beta_cos = math.cos(beta) 
        d = hamming_dist(i,j)
        return ((-1j * beta_sin)**d)*(beta_cos)**(self.__n - d)

    def U_gamma_entry(self, gamma, j):
        H_jj = H_jj_ring_fast(j, self.__n)
        return cmath.exp(-1j * gamma * H_jj)


    def psi_l(self, l, g1, Ub1, g2, Ub2):
        s = 0
        # t = time.time()
        for k in range(self.__N): # N = 2^n
            tk = time.time()
            k_in_n = ring.project(k,self.__n)
            if math.isnan(self.__ExpHam[k_in_n]):
                H_kk = H2_5[k_in_n]
                self.__ExpHam[k_in_n] = cmath.exp(-1j * H_kk)
            for j in range(self.__N):
                j_in_n = ring.project(j,self.__n)
                H_jj = H2_5[j_in_n]
                s = s + Ub2[l,k] * (self.__ExpHam[k_in_n]**g2) * Ub1[k,j] * cmath.exp(-1j * g1 * H_jj)
            print('Full k={} loop takes {}s'.format(k, time.time() - tk))
        return s / math.sqrt(self.__N)


    def build_psi_angled(self, gammas, betas):
        # Uc1 = expi(self.__Ham, gammas[0])
        Ub1 = expi(self.X_sum, betas[0])
        # Uc2 = expi(self.__Ham, gammas[1])
        Ub2 = expi(self.X_sum, betas[1])
        t = time.time()
        psi = [self.psi_l(k, gammas[0], Ub1, gammas[1], Ub2) for k in range(self.__N)]
        print('Full psi creation takes {}s'.format(time.time() - t))

        return psi
        
