import unittest
# import sys
from ringclean import * 
from LMC import * 
from numpy.linalg import norm
import numpy as np
import math
import concurrent.futures
# import matplotlib.pyplot as plt
# import pandas as pd
# import cmath
# from numpy import linalg as la
# import sympy
# from scipy.linalg import expm
# from scipy.linalg import block_diag
# from itertools import permutations
# from functools import reduce
import time
import csv
import time
from qutip import *
import multiprocessing.dummy as mp 
        
def ham_source_of_truth(n):
    diag = []
    for cut in range(2 ** n):
        cut = np.array(list(format(cut, 'b').zfill(n)), dtype=int)
        c = 1
        for i in range(n):
            nbd = find_nbhd(i, cut)
            if is_happy(nbd):
                # print(nbd)
                c = c + 1
                
        diag.append(c)
    return diag 


class TestRingClean(unittest.TestCase):

    def setUp(self):
        print("########################\nRunning method:{}\n########################".format(self._testMethodName))

    def test_easy_U_beta(self):
        n = 7
        B = X_sum(n)
        beta = 0.6
        U = expi(B, beta)

        alg = AlgInstance(n, [0.0], [beta])
        for row in range(2 ** n):
            for col in range(2 ** n):
                b = alg.Ub_el(row, col, 1)
                exp_b = U[row][col]
                self.assertTrue(np.allclose(b, exp_b), '{} should be equal to {}'.format(b, exp_b))


    def test_easy_cut_val(self):
        n = 7
        C = np.diag(ham_source_of_truth(n))
        gamma = 0.6

        alg = AlgInstance(n, [gamma], [0])
        for row in range(2 ** n):
            cv = alg.cut_value(row)
            exp_cv = C[row][row]
            self.assertTrue(np.allclose(cv, exp_cv), '{} should be equal to {}'.format(cv, exp_cv))

    def test_easy_U_gamma(self):
        n = 7
        C = np.diag(ham_source_of_truth(n))
        gamma = 0.6
        U = expi(C, gamma)

        alg = AlgInstance(n, [gamma], [0])
        for row in range(2 ** n):
            Ug_row = alg.Ug_el(row,1)
            # exp_cv = U[row][row]
            expe = U[row][row]
            self.assertTrue(np.allclose(Ug_row, expe), '{} should be equal to {}'.format(Ug_row, expe))


    def test_easy_U(self):
        n = 7
        C = np.diag(ham_source_of_truth(n))
        gamma = 0.6
        beta = 0.6
        Ug = expi(C, gamma)
        B = X_sum(n)
        Ub = expi(B, beta)

        alg = AlgInstance(n, [gamma], [beta])
        for x in range(2 ** n):
            for y in range(2 ** n):
                U_el = alg.U_el(x,y,1)
                # exp_cv = U[row][row]
                expe = Ug[x][x] * Ub[x][y]
                self.assertTrue(np.allclose(U_el, expe), '{} should be equal to {}'.format(U_el, expe))


    def xtest_easy_p1(self):
        n = 7
        g = 0.6
        b = 0.6

        alg = AlgInstance(n, [g], [b])
        ev = alg.run_p1()
        print(ev)

    def test_easy_p1_vec(self):
        n = 7
        g = 0.6
        b = 0.6

        alg = AlgInstance(n, [g], [b])
        ev = alg.run_p1_vec()
        print(ev)

    def xtest_p1_full(self):
        save = False

        n = 7
        num = 25

        print('#### Running QAOA for p=1, d=1 on size n={}'.format(n))
        best = -1

        if save:
            filename = 'n{}p1.csv'.format(n)
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['g', 'b', 'exp_val'])

        c = 1
        results = []
        t = time.time()
        for g in linspace(0.1, 2*pi, num = num):
            for b in linspace(0.1, pi, num = num):
                alg = AlgInstance(n, [g], [b])

                ev = alg.run_vec()
                ev2 = alg.run_p1_vec()
                self.assertTrue(np.allclose(ev, ev2), '{} should equal {} for angles {}, {}'.format(ev, ev2,g,b))

                results.append((g, b, ev))

                if ev > best:
                    best = ev
                    print('New best! ev={} ({}%), g={}, b={}'.format(best, best / n, g, b))

                if c % 100 == 0:
                    print('ev={} (total time: {}s)'.format(ev, time.time() - t))
                    if save:
                        print('saving! {}/{}'.format(c, num * num))
                        with open(filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            for res in results:
                                writer.writerow(res)

                    results = []
                
                c = c + 1

        print('Done! {}/{}'.format(c, num * num))
        if save:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for res in results:
                    writer.writerow(res)

        print('Took {}s'.format(time.time() - t))
        print('Overall best={} ({}%)'.format(best, best / n))

    def xtest_p2_full(self):
        save = False

        n = 11
        num = 10

        print('#### Running QAOA for p=2, d=1 on size n={}'.format(n))
        best = -1

        if save:
            filename = 'n{}p1.csv'.format(n)
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['g', 'b', 'exp_val'])

        c = 1
        results = []
        t = time.time()
        for g1 in linspace(0.1, 2*pi, num = num):
            for b1 in linspace(0.1, pi, num = num):
                for g2 in linspace(0.1, 2*pi, num = num):
                    for b2 in linspace(0.1, pi, num = num):
                        g = [g1, g2]
                        b = [b1, b2]
                        alg = AlgInstance(n, g, b)

                        # ev = alg.run_p1()
                        ev = alg.run_p2_vec()
                        # self.assertTrue(np.allclose(ev, ev2), '{} should equal {} for angles {}, {}'.format(ev, ev2,g1,b1))

                        results.append((g, b, ev))

                        if ev > best:
                            best = ev
                            print('New best! ev={} ({}%), g={}, b={}'.format(best, best / n, g, b))

                        if c % 100 == 0:
                            print('ev={} (total time: {}s)'.format(ev, time.time() - t))
                            if save:
                                print('saving! {}/{}'.format(c, num ** 4))
                                with open(filename, 'a', newline='') as csvfile:
                                    writer = csv.writer(csvfile)
                                    for res in results:
                                        writer.writerow(res)

                            results = []
                        
                        c = c + 1

        print('Done! {}/{}'.format(c, num * num))
        if save:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for res in results:
                    writer.writerow(res)

        print('Took {}s'.format(time.time() - t))
        print('Overall best={} ({}%)'.format(best, best / n))

    def test_p2_starting_good(self):
        save = False

        n = 11
        num = 3

        print('#### Running QAOA for p=2, d=1 on size n={}'.format(n))
        best = -1

        if save:
            filename = 'n{}p2.csv'.format(n)
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['g', 'b', 'exp_val'])

        c = 1
        results = []
        t = time.time()
        for g1 in linspace(0.1, 2*pi, num = num):
            for b1 in linspace(0.1, pi, num = num):
                g2 = 0.6152654422649655
                b2 = 1.8742623812607129
                g = [g1, g2]
                b = [b1, b2]
                alg = AlgInstance(n, g, b)

                ev = alg.run_vec()
                # self.assertTrue(np.allclose(ev, ev2), '{} should equal {} for angles {}, {}'.format(ev, ev2,g1,b1))

                results.append((g, b, ev))

                if ev > best:
                    best = ev
                    print('New best! ev={} ({}%), g={}, b={}'.format(best, best / n, g, b))

                if c % 100 == 0:
                    print('ev={} (total time: {}s)'.format(ev, time.time() - t))
                    if save:
                        print('saving! {}/{}'.format(c, num ** 2))
                        with open(filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            for res in results:
                                writer.writerow(res)

                    results = []
                
                c = c + 1

        print('Done! {}/{}'.format(c, num * num))
        if save:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for res in results:
                    writer.writerow(res)

        print('Took {}s'.format(time.time() - t))
        print('Overall best={} ({}%)'.format(best, best / n))

if __name__ == '__main__':
    unittest.main()
    
