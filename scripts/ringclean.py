import numpy as np
from math import sin, cos, sqrt, pi
from cmath import exp
from numpy import linspace
import csv
from numpy import linalg as LA

from time import time
from operator import mul
from functools import reduce
from itertools import combinations
from math import floor, ceil
import networkx as nx
import matplotlib.pyplot as plt
from pandas import DataFrame

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

def zifify(x):
    return 2* x - 1

def xifiy(z):
    return 1/2*(1+z) 

def f2(a,b,c):
    # H_b = a*(b + c) <= 0
    H_b = (a == b == c)
    # a = 1/2*(1 + a)
    # b = 1/2*(1 + b)
    # c = 1/2*(1 + c)

    # v = ((1 - b*a) + (1 - b*c) + (1-a*c)) / 4
    # v = a * b * c + (1-a)*(1-b)*(1-c)
    # v = 1/4*(1+a*b + b*c + a*c)
    v=1/4*(a*b + b*c + a*c +  1)
    # v = 1-(a+b+c) + (a*b + b*c + a*c)
    return(H_b, v)

def f3(a,b,c,d):
    H_a = a * (b+c+d) <= 0
    # v = ((1 - a*b) + (1 - a*c) + (1-a*d) - (1 - a*b*c*d)) / 4
    # v = 1-(1/2 + 1/4*a*(b+c+d-b*c*d))
    v = 1/2 - 1/4*a*(b+c+d - b*c*d)
    # v = a*(b+c+d) - a*b*c*d + b*c*d + a*c*d + a*b*d

    return (H_a, v)

def dumb_happy(a, nbrs):
    v0 = 0
    v1 = 0
    for nbr in nbrs:
        if nbr == 1: 
            v1 += 1
        else: 
            v0 += 1
    if a == 1:
        return v0 >= len(nbrs)/2.0
    else:
        return v1 >= len(nbrs)/2.0

# def f4(a,b,c,d,e):
def f_general(a,nbrs):
    combos = floor(len(nbrs) / 2.0) + 1
    combs = list(combinations(range(len(nbrs)), combos))

    H_a = dumb_happy(a,nbrs)

    # 0-1
    v1 = 0
    for indxs in combs:
        vinner = 1
        for idx in indxs:
            vinner = vinner * nbrs[idx]
        v1 = v1 + vinner
    v1 = v1*a

    v2 = 0
    for indxs in combs:
        vinner = 1
        for idx in indxs:
            vinner = vinner * (1-nbrs[idx])
        v2 = v2 + vinner
    v2 = v2*(1-a)

    v = v1+v2

    # Handle overcounted cases
    v = v - (len(combs) - 1)*(a*reduce(mul, nbrs, 1) + (1-a)*reduce(mul, [1-n for n in nbrs], 1))

    if len(nbrs) >= 5:
        # Handle undercounted cases
        x1 = 0
        x2 = 0
        for flipped in range(len(nbrs)):
            innerx1 = 1
            innerx2 = 1
            for idx in range(len(nbrs)):
                x3 = nbrs[idx]
                if idx == flipped:
                    x3 = 1-x3
                innerx1 = innerx1*(1-x3)
                innerx2 *= x3
            x1 += (1-a)*innerx1
            x2 += a*innerx2
        v = v -combos*(x1 + x2) #I dont think is quite general but it works

    v = 1-v
    # print('a={}, nbrs={}, v1={}, v2={}, happy={}, v={}'.format(a,nbrs,v1,v2,H_a,v))
    return (H_a, v)

# def calculate_fourier_coeffs(n):


def stringify(a,b,c,d=0,e=0,f=0,g=0,h=0):
    s = '('
    if a == -1: s += '~a*'
    else: s += 'a*'

    if b == -1:
        s += '~b*'
    else: s += 'b*'

    if c == -1:
        s += '~c'
    else: s += 'c'

    if d != 0:
        if d == -1:
            s += '*~d'
        else: s += '*d'

    if e != 0:
        if e == -1:
            s += '*~e'
        else: s += '*e'

    if f != 0:
        if f == -1:
            s += '*~f'
        else: s += '*f'

    if g != 0:
        if g == -1:
            s += '*~g'
        else: s += '*g'

    if h != 0:
        if h == -1:
            s += '*~h'
        else: s += '*h'
    s += ')'
    return s

def stringify_toZ(a,b,c,d=0,e=0,f=0,g=0,h=0):
    s = '('
    if a == -1: s += '-1/2*a*'
    else: s += '1/2(a-1)*'

    if b == -1:
        s += '-1/2*b*'
    else: s += '1/2(b-1)*'

    if c == -1:
        s += '-1/2*c'
    else: s += '1/2(c-1)'

    if d != 0:
        if d == -1:
            s += '*-1/2*d'
        else: s += '*1/2(d-1)'

    if e != 0:
        if e == -1:
            s += '*-1/2*e'
        else: s += '*1/2(e-1)'

    if f != 0:
        if f == -1:
            s += '*-1/2*f'
        else: s += '*1/2(f-1)'

    if g != 0:
        if g == -1:
            s += '*-1/2*g'
        else: s += '*1/2(g-1)'

    if h != 0:
        if h == -1:
            s += '*-1/2*h'
        else: s += '*1/2(h-1)'
    s += ')'
    return s



def nbhd_2_boolean():
    s = ''
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                    happy = a * (b+c) <= 0
                    if not happy:
                        s += stringify(a,b,c) + ' + '
    return s

def nbhd_3_boolean():
    s = ''
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                for d in [-1, 1]:
                        happy = a * (b+c+d) <= 0
                        if not happy:
                            s += stringify(a,b,c,d) + ' + '
    return s



def nbhd_4_boolean():
    s = ''
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                for d in [-1, 1]:
                    for e in [-1, 1]:
                        happy = a * (b+c+d+e) <= 0
                        if not happy:
                            s += stringify(a,b,c,d,e) + ' + '
    return s

def nbhd_5_boolean():
    s = ''
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                for d in [-1, 1]:
                    for e in [-1, 1]:
                        for f in [-1, 1]:
                            happy = a * (b+c+d+e+f) <= 0
                            if not happy:
                                s += stringify(a,b,c,d,e,f) + ' + '
    return s

def nbhd_6_boolean():
    s = ''
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                for d in [-1, 1]:
                    for e in [-1, 1]:
                        for f in [-1, 1]:
                            for g in [-1,1]:
                                happy = a * (b+c+d+e+f+g) <= 0
                                if not happy:
                                    s += stringify(a,b,c,d,e,f,g) + ' + '
    return s


def nbhd_7_boolean():
    s = ''
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                for d in [-1, 1]:
                    for e in [-1, 1]:
                        for f in [-1, 1]:
                            for g in [-1,1]:
                                for h in [-1,1]:
                                    happy = a * (b+c+d+e+f+g) <= 0
                                    if not happy:
                                        s += stringify(a,b,c,d,e,f,g) + ' + '
    return s

I = np.array([[1,0],[0,1]], int)
Z = np.array([[1,0],[0,-1]], int)

def Mi(i, n, M):
    individual_operators = [M if gate == i else I for gate in range(n)]
    return reduce(np.kron, individual_operators)

I_8 = Mi(0,8,I)



def to_int(v):
    return int(''.join([str(x) for x in v]), 2)

def construct_Hv(G, v):
    nbrs = G[v]
    print('{}, {}'.format(v,nbrs))

    if len(nbrs) == 3:
        return construct_Hv_d3(v, nbrs, G.order())
    else:
        raise Exception('Fuck you')

    Zv = Mi(v, G.number_of_nodes(), Z)
    Zis = [Mi(nbr, G.number_of_nodes(), Z) for nbr in nbrs]
    Hv = 1/2*(I_8 - 1/2*Zv @ (Zis[0] + Zis[1] + Zis[2] - Zis[0]@Zis[1]@Zis[2]))
    return Hv

def construct_Hv_d3(v, nbrs, n):

    Zv = Mi(v, n, Z)
    Zis = [Mi(nbr, n, Z) for nbr in nbrs]
    Hv = 1/2*(I_8 - 1/2*Zv @ (Zis[0] + Zis[1] + Zis[2] - Zis[0]@Zis[1]@Zis[2]))
    return Hv

def construct_H_cube():
    G = nx.hypercube_graph(3)

    # convert node labels to ints
    G = nx.relabel_nodes(G, lambda v : to_int(v)) 

    N = 2 ** G.order()
    # H = np.zeros( (N, N) )

    H = construct_Hv(G, 0)
    # for v in G.nodes():
    for v in range(1,G.order()):
        # print(v)
        Hv = construct_Hv(G, v)
        # H = and_hams(H, Hv)
        H = H + Hv

    # nx.draw_networkx(G)
    # plt.show()
    # print(DataFrame(H))
    # print(np.matrix(H))
    # Hm = np.matrix(H)
    diag = np.diagonal(H)
    for idx, d in enumerate(diag):
        cut = format(idx, '#010b')[2:]
        print(idx, d, cut)
        # if d == 0:
        #     print(idx)
    
    # nx.draw_networkx(G2)
    # plt.show()

def add_hams(H1, H2):
    return H1 + H2 - 2*H1 @ H2

def and_hams(H1, H2):
    return H1 @ H2

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
        state = self.build_state()
        return self.expectation(state)

    def build_state(self):
        temp_vec = np.ones(self.N) * self.c
        for p in range(1, len(self._g) + 1):
            temp_vec = self.apply_U_p(temp_vec, p)
        
        return temp_vec



    def run_vec_half(self):
        state = self.build_state_in_half()
        return self.expectation_half(state)

    def build_state_in_half(self):
        temp_vec = np.ones(int(self.N / 2)) * self.c
        for p in range(1, len(self._g) + 1):
            temp_vec = self.apply_U_p_half(temp_vec, p)

        return temp_vec

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

                b_val2 = self.Ub_el(k, j + size, p)
                g_val2 = self.Ug_el(j + size, p)
                s_k += b_val2 * g_val2 * vec[size - j - 1]
                

            temp_vec[k] = s_k 

        return temp_vec

    # <vec|C|vec>
    def expectation_half(self, vec):
        exp = 0
        size = int(self.N / 2)
        for k in range(size):
            exp += self.cut_value(k) * abs(vec[k]) ** 2
            # exp += self.cut_value(k) * abs(vec[size - k - 1]) ** 2

        # multiply by 2 to make up for only half counting
        return 2 * exp

    REPETITIONS = 4

    def apply_U_p(self, vec, p):
        temp_vec = np.zeros(self.N, dtype='complex')
        for k in range(self.N):
            # dumbm ap = {}
            s_k = 0
            # Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
            for j in range(self.N):
                # s_k = self.inner_U_p(vec, s_k, k, j, p)
                b_val = self.Ub_el(k, j, p)
                g_val = self.Ug_el(j, p)
                toadd= b_val * g_val * vec[j]
                # if toadd in dumbmap:
                #     dumbmap[toadd].append(j)
                # else:
                #     dumbmap[toadd] = [j]
                # print(toadd)
                s_k += toadd
            # print(dumbmap)
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

