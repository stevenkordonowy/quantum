import numpy as np
from math import sin, cos

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

def H_jj_ring_simple(cut, n):
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

def hamming_dist(n1, n2) : 

    x = n1 ^ n2  
    setBits = 0

    while (x > 0) : 
        setBits = setBits + x & 1
        x >>= 1
    
    return setBits  

class GammaGate:

    def __init__(self, g, n):
        self.gamma = g
        self.n = n

    def by_index(self, j, _):
        return H_jj_ring_simple(j, self.n)

class BetaGate:

    def __init__(self, b, n):
        self.beta = b
        self.n = n

    def by_index(self, j, k):
        beta_sin = sin(self.beta)
        beta_cos = cos(self.beta) 
        d = hamming_dist(j,k)
        return ((-1j * beta_sin)**d)*(beta_cos)**(self.n - d)


class AlgInstance:

    def __init__(self, n, k):
        self.n = n
        self.N = 2 ** n
        self.k = k


    def apply_single_gate_to_basis(self, C, i, t):
        vec = {}
        coeff_sq_sum = 0
        for k in range((t-1) * 2^(self.n - self.k)):
            v_k = C.by_index(k, i)
            vec[k: v_k]
            coeff_sq_sum = coeff_sq_sum + abs(v_k)^2
        
        return vec

def main():
    n = 5
    G = GammaGate(1.0, n)
    Gv = []
    for j in range(2**n):
        v = G.by_index(j,0)
        Gv.append(v)
        print('G[{}] = {}\n'.format(j, v))
    print(np.around(Gv,3))

if __name__ == "__main__":
    main()

