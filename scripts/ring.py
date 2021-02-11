import numpy as np

def ham_source_of_truth(n):
    diag = []
    for cut in range(2 ** n):
        cut = np.array(list(format(cut, 'b').zfill(n)), dtype=int)
        c = 0
        for i in range(n):
            nbd = find_nbhd(i, cut)
            if is_happy(nbd):
                # print(nbd)
                c = c + 1
                
        diag.append(c)
    return diag 

def construct_ham_local(vtx, dist):
    diag = []
    vtx_nbhd = nbhd_vtxs(vtx, dist, 2 * dist + 1)
    n = 2 * dist + 1
    for cut in range(2 ** n):
        cut = np.array(list(format(cut, 'b').zfill(n)), dtype=int)
        c = 0
        for inside in vtx_nbhd:
            nbd = find_nbhd(inside, cut)
            if is_happy(nbd):
                c = c + 1
                
        diag.append(c)
    return np.diag(diag)

def construct_full_ham2(n):
    H = construct_ham_local2(0,n)
    for i in range(1,n):
        H = H + construct_ham_local2(i,n)
    return H / 3

def construct_ham_local2(vtx, n):
    diag = []
    vtx_nbhd = nbhd_vtxs(vtx, 1, n)
    for cut in range(2 ** n):
        cut = np.array(list(format(cut, 'b').zfill(n)), dtype=int)
        c = 0
        for inside in vtx_nbhd:
            nbd = find_nbhd(inside, cut)
            if is_happy(nbd):
                c = c + 1
                
        diag.append(c)
    return np.diag(diag)

def construct_ham_local2_as_arr(vtx, n):
    diag = []
    vtx_nbhd = nbhd_vtxs(n - vtx - 1, 1, n)
    for cut in range(2 ** n):
        cut = np.array(list(format(cut, 'b').zfill(n)), dtype=int)
        c = 0
        for inside in vtx_nbhd:
            nbd = find_nbhd(inside, cut)
            if is_happy(nbd):
                c = c + 1
            # else:
            #     print('not happy: cut={}, nbd={}, idx={}'.format(cut, nbd, inside))
                
        diag.append(c)
    return diag

def is_happy(nbhd):
    return not (nbhd[0] == nbhd[1] == nbhd[2])

def find_nbhd(i, s):
    indices = range(i-1, i+2)
    return s.take(indices, mode='wrap')


def nbhd_vtxs(vtx, dist, n):
    indices = range(vtx-dist, vtx+dist+1)
    return np.array(list(range(n))).take(indices, mode='wrap')        


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

        