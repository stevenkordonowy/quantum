import numpy as np
from math import sin, cos, sqrt, pi
from numpy import linspace
import csv
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
from functools import reduce
from scipy.linalg import expm

def fast_ev_d2(g,b):
    sin_g = sin(g)
    sin_g_half = sin(g/2)
    cos_g = cos(g)
    cos_g_half = cos(g/2)

    sin_2b = sin(2*b)
    cos_2b = cos(2*b)

    v1 = cos_2b*sin_2b*sin_g*cos_g*cos_g_half**2
    v2 = sin_2b**2 * sin_g_half*sin_g*cos_g_half**3 * cos_g
    v3 = 2*cos_2b*sin_2b*sin_g_half*cos_g_half*cos_g**2
    v4 = sin_2b**2 * sin_g**2*cos_g**2*cos_g_half**2

    return 3/4 - v1-v2-1/4*(v3+v4)

def fast_ev_d3(g, b):
    sin_g = sin(g)
    sin_g_half = sin(g/2)
    cos_g = cos(g)
    cos_g_half = cos(g/2)

    sin_2b = sin(2*b)
    cos_2b = cos(2*b)
    
    v1 = cos_2b * sin_2b*(sin_g_half * sin_g ** 2 * cos_g_half**3 * cos_g -sin_g*cos_g**2 * cos_g_half**4)
    v2 = cos_2b**3 * sin_2b*sin_g_half * cos_g**3 * cos_g_half**3
    v3 = cos_2b*sin_2b**3*(sin_g_half*cos_g**5 * cos_g_half**3 + sin_g_half*cos_g_half**9 * cos_g**9 + sin_g**3 * cos_g**6*cos_g_half**10 + sin_g_half**3*sin_g**6 * cos_g_half**7 * cos_g**3)

    return 1/2 - 3/2 * v1 + v2 + 1/4 * v3

def pprint(res):
    for r in res:
        print('{}, {}, {}'.format(r[0], r[1], r[2]))


def run_d2(num = 100):
    res = []

    for g in linspace(0, pi/4, num = num):
        for b in linspace(0, pi/2, num = num):
            res.append((g,b,fast_ev_d2(g,b)))

    # pprint(res)
    res = np.array(res)
    with open('d2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['g', 'b', 'exp_val'])
        for r in res:
            writer.writerow(r)

    best = max(res, key=itemgetter(2))
    print('Best: {}, {}, {}'.format(best[0], best[1], best[2]))

    return res

def run_d3(num = 100):
    res = []

    for g in linspace(0, pi/4, num = num):
        for b in linspace(0, pi/2, num = num):
            res.append((g,b,fast_ev_d3(g,b)))

    # pprint(res)
    res = np.array(res)
    with open('d3.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['g', 'b', 'exp_val'])
        for r in res:
            writer.writerow(r)

    best = max(res, key=itemgetter(2))
    print('Best: {}, {}, {}'.format(best[0], best[1], best[2]))

    return res

def run_sims_and_plot():
    d2_res = run_d2()
    d3_res = run_d3()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmhot = plt.get_cmap("BuPu")
    ax.scatter(d2_res[:,0], d2_res[:,1], d2_res[:,2],c=d2_res[:,2], cmap=cmhot, label='d2')
    YlGn = plt.get_cmap('YlGn')
    ax.scatter(d3_res[:,0], d3_res[:,1], d3_res[:,2], c=d3_res[:,2], cmap=YlGn, label='d3')
    plt.legend(loc='upper left')
    # plt.show()

    gamma_d2 = 0.5711986642890533
    beta_d2 = 2.824260062318097
    gamma_d3 = 0.4442656277803748
    beta_d3 = 1.9357288067573473

    d2_best = fast_ev_d2(gamma_d2,beta_d2)
    d3_best = fast_ev_d3(gamma_d3,beta_d3)
    d2_using_d3best = fast_ev_d2(gamma_d3,beta_d3)
    d2_using_d2best = fast_ev_d3(gamma_d2,beta_d2)
    print('{}, {}, {}, {}'.format(d2_best, d2_using_d3best, d3_best, d2_using_d2best))

from itertools import chain, combinations

# Ignores empty set
def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return set(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))

subs = powerset([2, 5])
print(subs)

# d = 2
def construct_MM(n):
    MM = []
    for i in range(n):
        edge = frozenset({i, (i + 1) % n})
        MM.append(edge)
    for i in range(n):
        edge = frozenset({i, (i + 2) % n})
        MM.append(edge)
    return MM

MM = construct_MM(7)
print(MM)

from quantum_helpers import *

I_8 = M_K({0},8,I)

edges = []
for i in range(7):
    edge = frozenset({i, (i + 1) % 7})
    edges.append(edge)

nonedges = []
for i in range(7):
    nonedge = frozenset({i, (i + 2) % 7})
    nonedges.append(nonedge)

# d = 2
def W_M(K):
    if K in edges:
        return -1/2
    elif K in nonedges:
        return -1/4
    
    return 0

# d = 2
def zk_expectation(K, n, g, b):
    Z_K = M_K(K,n,Z)

    subsets = powerset(K)
    MM = construct_MM(n)

    operator_summed = np.zeros((2**n, 2**n), dtype='complex128')
    for L in subsets:
        X_L = M_K(L, n, X)

        # Construct product
        prod = M_K(set(),n,I)
        for M in MM:
            LintM = set(L).intersection(M)
            if len(LintM) % 2 == 1:
                w = W_M(M)
                angle = -2j * g * w
                Z_M = M_K(M, n, Z)
                e_ZM = expm(angle * Z_M)
                prod = prod @ e_ZM

        inner_operator = X_L @ prod
        operator_summed += 1j**len(L) * cos(2*b)**(len(K) - len(L)) * sin(2*b)**len(L) * inner_operator

    s = plus_n(n)
    return np.transpose(s) @ Z_K @ operator_summed @ s


def full_expectation(n, g, b):
    v1 = 3*n/4
    v2 = 0
    for v in range(n):
        v2 += zk_expectation([v, (v + 1) % n], n, g, b)

    v3 = 0
    for v in range(n):
        v3 += zk_expectation([v, (v + 2) % n], n, g, b)

    return v1 -1/2*v2 - 1/4*v3


r = zk_expectation(set([0,1]), 7, pi, pi)
print(round(r[0][0], 20))


gamma_d2 = 0.5711986642890533
beta_d2 = 2.824260062318097

d2_best = fast_ev_d2(gamma_d2,beta_d2)
ev2 = full_expectation(7, gamma_d2,beta_d2)[0][0]
print(d2_best, round(ev2,15) / 7)
