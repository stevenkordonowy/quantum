import numpy as np
from math import sin, cos, sqrt, pi
from cmath import exp
from numpy import linspace
import csv
from numpy import linalg as LA

from time import time


# I = AlgInstance(15, 5)


def nbhd_vtxs(vtx, dist, n):
    indices = range(vtx-dist, vtx+dist+1)
    return np.array(list(range(n))).take(indices, mode='wrap')  


def find_nbhd(i, s):
    indices = range(i-1, i+2)
    return s.take(indices, mode='wrap')


def is_happy(nbhd):
    return not (nbhd[0] == nbhd[1] == nbhd[2])


def ring_local_ham(vtx, n):
    diag = []
    vtx_nbhd = nbhd_vtxs(n - vtx - 1, 1, n)
    for cut in range(2 ** n):
        cut = np.array(list(format(cut, 'b').zfill(n)), dtype=int)
        c = 0
        for inside in vtx_nbhd:
            nbd = find_nbhd(inside, cut)
            if is_happy(nbd):
                # print(nbd)
                c = c + 1
                
        diag.append(c)
    return diag

H2_5 = ring_local_ham(2, 5)

def cut_value(cut, n):
    if n == 5:
        return H_jj_ring_fast_5(cut)
    
    cut_in_5 = project(cut, n)
    num_happy = H2_5[cut_in_5]

    for k in range(1, n):
        cut_in_5 = shift(cut, k, n)
        rotated_j_in_5 = project(cut_in_5, n)
        num_happy = num_happy + H2_5[rotated_j_in_5]
    return num_happy / 3


def H_jj_ring_fast_5(cut):
    num_happy = H2_5[cut]

    for k in range(1, 5):
        cut_in_5 = shift(cut, k, 5)
        num_happy = num_happy + H2_5[cut_in_5]
    return num_happy / 3

'''
 Shifts a bitstring (really an integer) 'cut' left 'reps' times
'''
def shift(cut, reps, dim):
    x = _shift(cut,dim)
    for _ in range(reps-1):
        x = _shift(x,dim)
    return x

'''
Considers 'cut' as a binary string of length 'dim' and logically
shifts it left. Ie 01001 -> 10010. I say 'logically' because
'cut' is an integer not a bitstring, so 'dim' is required to properly rotate
'''
def _shift(cut, dim):
    if cut  < 2**(dim - 1):
        return int(2*cut)
    else:
        return int(2*cut + 1 - 2**dim)

def project(cut, frum):
    new_cut = project_1(cut, frum)
    for frum_ in range(frum-1, 5, -1):
        new_cut = project_1(new_cut, frum_)
    return new_cut


def project_1(cut, frum):
    lim = 2**(frum-1)
    if cut < lim:
        return cut
    else:
        return cut - lim


def hamming_dist(n1, n2):
    x = n1 ^ n2  
    setBits = 0

    while (x > 0) : 
        setBits += x & 1
        x >>= 1
    
    return setBits  


class AlgInstance:

    def __init__(self, n, g, b):
        self.n = n

        if len(g) != len(b):
            raise ValueError('g length {} does not match b length P{}'.format(len(g), len(b)))
        elif len(g) == 0:
            raise ValueError('Cannot handle p=0')

        self._g = g
        self._b = b

        self.N = 2 ** n
        self.c = 1 / sqrt(self.N)
        self.cut_cache = {} # Always use this, space is O(2**n)
        self.gamma_cache = {} # Always use this, space is O(2**n)

        self.beta_cache = {}
        self.full_cache = {}
        self.cache = True


    def run_vec(self):
        temp_vec = np.ones(self.N) * self.c
        for p in range(1, len(self._g) + 1):
            temp_vec = self.apply_U_p(temp_vec, p)

        return self.expectation(temp_vec)

    def run_vec_in_half(self):
        temp_vec = np.ones(int(self.N / 2)) * self.c
        for p in range(1, len(self._g) + 1):
            temp_vec = self.apply_U_p_half(temp_vec, p)

        return self.expectation(np.concatenate((temp_vec, np.flip(temp_vec))))

    def run_vec_half(self):
        temp_vec = np.ones(int(self.N / 2)) * self.c
        for p in range(1, len(self._g) + 1):
            temp_vec = self.apply_U_p_half(temp_vec, p)

        return self.expectation(np.concatenate((temp_vec, np.flip(temp_vec))))

    def apply_U_p_half(self, vec, p):
        size = int(self.N / 2)
        temp_vec = np.zeros(size, dtype='complex')
        for k in range(size):
            s_k = 0
            # Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
            for j in range(size):
                # s_k = self.inner_U_p(vec, s_k, k, j, p)
                b_val = self.Ub_el(k, j, p)
                g_val = self.Ug_el(j, p)
                s_k += b_val * g_val * vec[j]

            temp_vec[k] = s_k

        return temp_vec

    # <vec|C|vec>
    def expectation_half(self, vec):
        exp = 0
        for k in range(int(self.N / 2)):
            exp += self.cut_value(k) * abs(vec[k]) ** 2

        return 2 * sqrt(2) * exp

    # def run_

    REPETITIONS = 4

    def apply_U_p(self, vec, p):
        temp_vec = np.zeros(self.N, dtype='complex')
        for k in range(self.N):
            s_k = 0
            # Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
            for j in range(self.N):
                # s_k = self.inner_U_p(vec, s_k, k, j, p)
                b_val = self.Ub_el(k, j, p)
                g_val = self.Ug_el(j, p)
                s_k += b_val * g_val * vec[j]

            temp_vec[k] = s_k

        return temp_vec
        

    def inner_U_p(self, vec, curr_sum, k, j, p):
        b_val = self.Ub_el(k, j, p)
        g_val = self.Ug_el(j, p)
        return curr_sum + b_val * g_val * vec[j]

    # <vec|C|vec>
    def expectation(self, vec):
        exp = 0
        for k in range(self.N):
            exp += self.cut_value(k) * abs(vec[k]) ** 2

        return exp


    # <x|U|y> = <x|Ub Ug|y> = <x|Ub|y> <y|Ug|y>
    def U_el(self,x,y,p):
        b = self.Ub_el(x,y,p)
        g = self.Ug_el(x,p)
        toret = g * b
        # print('<{}|U|{}>={} (g={},b={})'.format(x,y,toret,g,b))
        return toret


    # <x|C|x>
    def cut_value(self, x):
        if x not in self.cut_cache:
            self.cut_cache[x] = cut_value(x, self.n)
        return self.cut_cache[x]

    # <x|Ug|x> = <x|exp(-i*gamma*C)|x>
    def Ug_el(self, x, p):
        if (x,p) not in self.gamma_cache:
            g = self.g(p)
            cv = self.cut_value(x)
            self.gamma_cache[(x,p)] = exp(-1j * g * cv)
        return self.gamma_cache[(x,p)]

    # <x|Ub|y> = <x|exp(-i*beta*Xn)|x>
    def Ub_el(self, x, y, p):
        d = hamming_dist(x, y)
        beta = self.b(p)

        if self.cache:
            if (d,p) not in self.beta_cache:
                self.beta_cache[(d,p)] = ( (-1j * sin(beta)) ** d ) * cos(beta)**(self.n - d)
            return self.beta_cache[(d,p)]
        else:
            return ( (-1j * sin(beta)) ** d ) * cos(beta)**(self.n - d)


    def b(self, p):
        return self._b[p-1]

    def g(self, p):
        return self._g[p-1]


# def main():
    # num = 250
    # run_p1(7, num)
    # run_p2(11,num)
    


# if __name__ == "__main__":
#     main()

