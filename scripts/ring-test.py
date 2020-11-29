import unittest
# import sys
from ring import * 
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
# import time
import csv
import time
from qutip import *
import multiprocessing.dummy as mp 

def write_results(n, results, name):
    pd.DataFrame(results).to_csv("{0}-{1}.csv".format(n, name), header=None, index=None)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
    
def create_or_get_gamma_unitary(unitaries, angle, n):
    if angle in unitaries.keys():
        return unitaries[angle]
    else:
        U = construct_U_gamma(angle, n)
        unitaries[angle] = U
        return U

def create_or_get_beta_unitary(unitaries, angle, n):
    if angle in unitaries.keys():
        return unitaries[angle]
    else:
        U = construct_U_beta(angle, n)
        unitaries[angle] = U
        return U 

def create_or_get_gamma_unitary_qutip(unitaries, angle, n):
    if angle in unitaries.keys():
        return unitaries[angle]
    else:
        U = Qobj(construct_U_gamma_qutip(angle, n), copy=False)
        unitaries[angle] = U
        return U

def create_or_get_beta_unitary_qutip(unitaries, angle, n):
    if angle in unitaries.keys():
        return unitaries[angle]
    else:
        U = Qobj(construct_U_beta(angle, n), copy=False)
        unitaries[angle] = U
        return U 

def run_thing(lmc, U, unitaries_beta, b2, n, exp_val):
    U_b2 = create_or_get_beta_unitary(unitaries_beta, b2, n)
    
    # U = Ub2 * Ug2 * Ub1 * Ug1
    U = U_b2.dot(U)

    psi = U.dot(lmc.plus_n)
    exp_val = psi.conj().T.dot(lmc.Ham).dot(psi)
    exp_val = abs(exp_val.item())
        

class TestRing(unittest.TestCase):

    def setUp(self):
        print("########################\nRunning method:{}\n########################".format(self._testMethodName))


    def xtest_full_angles_p1(self):
        n = 7
        with open('n7p1.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['g1', 'b1', 'exp_val'])

        tham = time.time()
        C = np.diag(ham_source_of_truth(n))
        lmc = LocalMaxCut(C)
        print('{}s for ham'.format(time.time() - tham))

        running_best = (0.0,)
        results = []
        c1 = 1
        num_gamma = 250
        num_beta = 250
        unitaries_gamma = {}
        unitaries_beta = {}
        for g1 in np.linspace(0.1, math.pi, num = num_gamma):
            U_g1 = create_or_get_unitary(unitaries_gamma, g1, lmc.Ham)
            for b1 in np.linspace(0.1,  math.pi/2, num = num_beta):
                t = time.time()
                U_b1 = create_or_get_unitary(unitaries_beta, b1, lmc.X_sum)
        
                exp_val = lmc.schrodinger_ev([U_g1], [U_b1])
                results.append((g1, b1,exp_val))
                if exp_val > running_best[0]:
                    running_best = (exp_val, exp_val/n, g1, b1)
                    print("{0},{1},{2},{3},{4}".format(n, *running_best))
                if c1 % 100 == 0:
                    print('schrodinger ev={}({}%) ({}s)'.format(exp_val, exp_val / n, time.time() - t))
                    print('saving! {}'.format(c1))
                    with open('n7p1.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for res in results:
                            writer.writerow(res)

                    results = []

                c1 = c1 + 1


    def xtest_sp_angles_p1(self):
        n = 7
        filename = 'n7p1-sp.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['g1', 'b1', 'exp_val'])

        tham = time.time()
        C = np.diag(ham_source_of_truth(n))
        lmc = LocalMaxCut(C)
        print('{}s for ham'.format(time.time() - tham))

        running_best = (0.0,)
        results = []
        c1 = 1
        num_gamma = 1000
        num_beta = 100
        total = num_gamma * num_beta
        g = 0.5973608751237137
        b = 0.31433428808887287
        curr_best = 0.9393746244466347
        unitaries_gamma = {}
        unitaries_beta = {}
        for g1 in np.linspace(g * 0.999, g * 1.001, num = num_gamma):
            U_g1 = create_or_get_unitary(unitaries_gamma, g1, lmc.Ham)
            for b1 in np.linspace(b * 0.999, b * 1.001, num = num_beta):
                t = time.time()
                U_b1 = create_or_get_unitary(unitaries_beta, b1, lmc.X_sum)
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lmc.schrodinger_ev, [U_g1], [U_b1] )
                    # _thread.start_new_thread(lmc.schrodinger_ev, ([U_g1], [U_b1], ) )
                    exp_val = future.result()

                # exp_val = lmc.schrodinger_ev([U_g1], [U_b1])
                results.append((g1, b1,exp_val))
                if exp_val > running_best[0]:
                    running_best = (exp_val, exp_val/n, g1, b1)
                    print("{0},{1},{2},{3},{4}".format(n, *running_best))

                if running_best[1] > curr_best:
                    print("New best!!! {}".format(*running_best))
                    
                if c1 % 100 == 0:
                    print('schrodinger ev={}({}%) ({}s)'.format(exp_val, exp_val / n, time.time() - t))
                    print('saving! {}/{}'.format(c1, total))
                    with open(filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for res in results:
                            writer.writerow(res)

                    results = []

                c1 = c1 + 1

        print("BEST: {0},{1},{2},{3},{4}".format(n, *running_best))


    def xtest_full_angles_p2(self):
        n = 11
        filename = 'n11p2-full.csv'
        # with open(filename, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(['g1', 'b1', 'g1', 'b1', 'exp_val'])

        tham = time.time()
        C = np.diag(ham_source_of_truth(n))
        lmc = LocalMaxCut(C)
        print('{}s for ham'.format(time.time() - tham))

        running_best = (0.0,)
        results = []
        c1 = 1
        num_gamma = 3
        num_beta = 3
        unitaries_gamma = {}
        unitaries_beta = {}
        total_time = time.time()
        for g1 in np.linspace(0.1, math.pi, num = num_gamma):
            U_g1 = create_or_get_gamma_unitary(unitaries_gamma, g1, n)
            for b1 in np.linspace(0.1,  math.pi/2, num = num_beta):
                U_b1 = create_or_get_beta_unitary(unitaries_beta, b1, n)
                
                for g2 in np.linspace(math.pi/2, 3*math.pi/2, num = num_gamma):
                    U_g2 = create_or_get_gamma_unitary(unitaries_gamma, g2, n)
                    for b2 in np.linspace(math.pi/4, 3*math.pi/4, num = num_beta):
                        t = time.time()
                        U_b2 = create_or_get_beta_unitary(unitaries_beta, b2, n)
                        
                        exp_val = lmc.schrodinger_ev([U_g1, U_g2], [U_b1, U_b2])
                        results.append((g1, b1, g2, b2, exp_val))
                        if exp_val > running_best[0]:
                            running_best = (exp_val, exp_val/n, g1, b1, g2, b2)
                            print("{0},{1},{2},{3},{4},{5},{6}".format(n, *running_best))
                        if c1 % 100 == 0:
                            print('ev={} (total loop time: {}s)'.format(exp_val, time.time() - t))
                            # print('saving! {}'.format(c1))
                            # with open(filename, 'a', newline='') as csvfile:
                            #     writer = csv.writer(csvfile)
                            #     for res in results:
                            #         writer.writerow(res)

                            results = []

                        c1 = c1 + 1
        print("Overall time: {}\nBEST: {},{},{},{},{}".format(time.time() - total_time, n, *running_best))

    def xtest_full_angles_p2_multiply_as_we_go(self):
        n = 11
        filename = 'n11p2-full.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['g1', 'b1', 'g1', 'b1', 'exp_val'])

        tham = time.time()
        C = np.diag(ham_source_of_truth(n))
        lmc = LocalMaxCut(C)
        print('{}s for ham'.format(time.time() - tham))

        running_best = (0.0,)
        results = []
        c1 = 1
        num_gamma = 3
        num_beta = 3
        unitaries_gamma = {}
        unitaries_beta = {}
        total_time = time.time()

        p=mp.Pool(4)

        for g1 in np.linspace(0.1, math.pi, num = num_gamma):
            t = time.time()
            U_g1 = create_or_get_gamma_unitary(unitaries_gamma, g1, n)
            t1 = time.time() - t
            print('U_g1 took {}s'.format(t1))
            U = U_g1
            for b1 in np.linspace(0.1,  math.pi/2, num = num_beta):
                t = time.time()
                U_b1 = create_or_get_beta_unitary(unitaries_beta, b1, n)
                t1 = time.time() - t
                print('U_b1 took {}s'.format(t1))
                U = U_b1.dot(U)
                for g2 in np.linspace(math.pi/2, 3*math.pi/2, num = num_gamma):
                    t = time.time()
                    U_g2 = create_or_get_gamma_unitary(unitaries_gamma, g2, n)
                    U = U_g2.dot(U)

                    p.map(create_or_get_gamma_unitary,np.linspace(math.pi/4, 3*math.pi/4, num = num_beta)) # range(0,1000) if you want to replicate your example
                    p.close()
                    p.join()
                    for b2 in np.linspace(math.pi/4, 3*math.pi/4, num = num_beta):
                        t = time.time()
                        exp_val = 0
                        run_thing(lmc, U, unitaries_beta, b2, n, exp_val)

                        # results.append(result)
                        if exp_val > running_best[0]:
                            running_best = (exp_val, exp_val/n, g1, b1, g2, b2)
                            print("{0},{1},{2},{3},{4},{5},{6} [{7}s]".format(n, *running_best, t1))
                        if c1 % 100 == 0:
                            print('ev={} (total loop time: {}s)'.format(exp_val, t1))
                            print('saving! {}'.format(c1))
                            with open(filename, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                for res in results:
                                    writer.writerow(res)

                            results = []

                        c1 = c1 + 1
        print("Overall time: {}\nBEST: {},{},{},{},{}".format(time.time() - total_time, n, *running_best))

    def xtest_full_angles_p2_multiply_as_we_go_qutip(self):
        n = 11
        filename = 'n11p2-BLAH.csv'
        # with open(filename, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(['g1', 'b1', 'g1', 'b1', 'exp_val'])

        tham = time.time()
        C = np.diag(ham_source_of_truth(n))
        lmc = LocalMaxCut(C)
        print('{}s for ham'.format(time.time() - tham))
        H = Qobj(lmc.Ham)

        running_best = (0.0,)
        results = []
        c1 = 1
        num_gamma = 3
        num_beta = 3
        unitaries_gamma = {}
        unitaries_beta = {}
        total_time = time.time()
        for g1 in np.linspace(0.1, math.pi, num = num_gamma):
            t = time.time()
            U_g1 = create_or_get_gamma_unitary_qutip(unitaries_gamma, g1, n)
            t1 = time.time() - t
            print('U_g1 took {}s'.format(t1))
            U = U_g1
            for b1 in np.linspace(0.1,  math.pi/2, num = num_beta):
                U_b1 = create_or_get_beta_unitary_qutip(unitaries_beta, b1, n)
                U = U_b1 * U
                for g2 in np.linspace(math.pi/2, 3*math.pi/2, num = num_gamma):
                    t = time.time()
                    U_g2 = create_or_get_gamma_unitary_qutip(unitaries_gamma, g2, n)
                    U = U_g2 * U
                    for b2 in np.linspace(math.pi/4, 3*math.pi/4, num = num_beta):
                        t = time.time()
                        U_b2 = create_or_get_beta_unitary_qutip(unitaries_beta, b2, n)
                        
                        # U = Ub2 * Ug2 * Ub1 * Ug1
                        U = U_b2 * U
                        psi = U*lmc.plus_n

                        exp_val = expect(H, Qobj(psi))
                        t1 = time.time() - t

                        print('ev={} (total loop time: {}s)'.format(exp_val, t1))

                        results.append((g1, b1, g2, b2, exp_val))
                        if exp_val > running_best[0]:
                            running_best = (exp_val, exp_val/n, g1, b1, g2, b2)
                            print("{0},{1},{2},{3},{4},{5},{6} [{7}s]".format(n, *running_best, t1))
                        if c1 % 100 == 0:
                            print('ev={} (total loop time: {}s)'.format(exp_val, t1))
                            # print('saving! {}'.format(c1))
                            # with open(filename, 'a', newline='') as csvfile:
                            #     writer = csv.writer(csvfile)
                            #     for res in results:
                            #         writer.writerow(res)

                            results = []

                        c1 = c1 + 1
        print("Overall time: {}\nBEST: {},{},{},{},{}".format(time.time() - total_time, n, *running_best))

    def xtest_specific_angles_p2(self):
        n = 11
        filename = 'n11p2-sp.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['g1', 'b1', 'g1', 'b1', 'exp_val'])

        C = np.diag(ham_source_of_truth(n))
        lmc = LocalMaxCut(C)

        running_best = (0.0,)
        results = []
        c1 = 1
        num_gamma = 10
        num_beta = 10
        l_mult = 0.95
        u_mult = 1.05

        G1 = 0.5973608751237137
        B1 = 0.31433428808887287
        G2 = 4.581489286485114
        B2 = 1.4398966328953218

        unitaries_gamma = {}
        unitaries_beta = {}
        for g1 in np.linspace(l_mult * G1, u_mult*G1, num = num_gamma):
            U_g1 = create_or_get_unitary(unitaries_gamma, g1, lmc.Ham)
            for b1 in np.linspace(l_mult * B1, u_mult*B1, num = num_beta):
                U_b1 = create_or_get_unitary(unitaries_beta, b1, lmc.X_sum)
                for g2 in np.linspace(l_mult * G2, u_mult*G2, num = num_gamma):
                    U_g2 = create_or_get_unitary(unitaries_gamma, g2, lmc.Ham)
                    for b2 in np.linspace(l_mult * B2, u_mult*B2, num = num_beta):
                        t = time.time()
                        U_b2 = create_or_get_unitary(unitaries_beta, b2, lmc.X_sum)

                        exp_val = lmc.schrodinger_ev([U_g1, U_g2], [U_b1, U_b2])
                        results.append((g1, b1, g2, b2, exp_val))
                        if exp_val > running_best[0]:
                            running_best = (exp_val, exp_val/n, g1, b1, g2, b2)
                            print("{0},{1},{2},{3},{4},{5},{6}".format(n, *running_best))
                        if c1 % 100 == 0:
                            print('ev={} (total loop time: {}s)'.format(exp_val, time.time() - t))
                            print('saving! {}'.format(c1))
                            with open(filename, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                for res in results:
                                    writer.writerow(res)

                            results = []

                        c1 = c1 + 1
        
        print("BEST: {0},{1},{2},{3},{4}".format(n, *running_best))


    def xtest_half_angles_p2(self):
        n = 11
        filename = 'n11p2-half-centered-flipped.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['g1', 'b1', 'g1', 'b1', 'exp_val'])

        tham = time.time()
        C = np.diag(ham_source_of_truth(n))
        lmc = LocalMaxCut(C)
        print('{}s for ham'.format(time.time() - tham))

        running_best = (0.0,)
        results = []
        c1 = 1
        num_gamma = 70
        num_beta = 70
        g = 0.5973608751237137
        b = 0.31433428808887287

        G2 = 4.581489286485114
        B2 = 1.4398966328953218
        curr_best = 0.9393746244466347
        unitaries_gamma = {}
        unitaries_beta = {}
        U_g1 = create_or_get_unitary(unitaries_gamma, G2, lmc.Ham)
        U_b1 = create_or_get_unitary(unitaries_beta, B2, lmc.X_sum)

        l_mult = 0.9
        u_mult = 1.1
        for g2 in np.linspace(l_mult * g, u_mult*g, num = num_gamma):
            U_g2 = create_or_get_unitary(unitaries_gamma, g2, lmc.Ham)
            for b2 in np.linspace(l_mult * b, u_mult*b, num = num_beta):
                t = time.time()
                U_b2 = create_or_get_unitary(unitaries_beta, b2, lmc.X_sum)
                
                exp_val = lmc.schrodinger_ev([U_g1, U_g2], [U_b1, U_b2])
                results.append((g, b, g2, b2, exp_val))
                if exp_val > running_best[0]:
                    running_best = (exp_val, exp_val/n, g, b, g2, b2)
                    print("{0},{1},{2},{3},{4},{5},{6}".format(n, *running_best))
                if c1 % 100 == 0:
                    print('ev={} (total loop time: {}s)'.format(exp_val, time.time() - t))
                    print('saving! {}'.format(c1))
                    with open(filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for res in results:
                            writer.writerow(res)

                    results = []

                c1 = c1 + 1

        print("BEST: {0},{1},{2},{3},{4}".format(n, *running_best))


    def xtest_full_angles_p3(self):
        n = 15
        filename = 'n15p3-full.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['g1', 'b1', 'g2', 'b2', 'g3', 'b3' 'exp_val'])

        tham = time.time()
        C = np.diag(ham_source_of_truth(n))
        print('{}s for ham'.format(time.time() - tham))
        lmc = LocalMaxCut(C)

        running_best = (0.0,)
        results = []
        c1 = 0
        num_gamma = 10
        num_beta = 10
        unitaries_gamma = {}
        unitaries_beta = {}
        total = 10**6
        for g1 in np.linspace(0.1, math.pi, num = num_gamma):
            U_g1 = create_or_get_gamma_unitary(unitaries_gamma, g1, n)
            U = U_g1
            for b1 in np.linspace(0.1,  math.pi/2, num = num_beta):
                U_b1 = create_or_get_beta_unitary(unitaries_beta, b1, n)
                U = U_b1.dot(U)
                for g2 in np.linspace(0.01, math.pi, num = num_gamma):
                    U_g2 = create_or_get_gamma_unitary(unitaries_gamma, g2, n)
                    U = U_g2.dot(U)
                    for b2 in np.linspace(0.01,  math.pi/2, num = num_beta):
                        U_b2 = create_or_get_beta_unitary(unitaries_beta, b2, n)
                        U = U_b2.dot(U)
                        for g3 in np.linspace(0.02, math.pi, num = num_gamma):
                            U_g3 = create_or_get_gamma_unitary(unitaries_gamma, g3, n)
                            U = U_g3.dot(U)
                            for b3 in np.linspace(0.02,  math.pi/2, num = num_beta):
                                t = time.time()
                                U_b3 = create_or_get_beta_unitary(unitaries_beta, b3, lmc.X_sum)

                                # U = Ub3 * Ug3 * Ub2 * Ug2 * Ub1 * Ug1
                                U = U_b3.dot(U)
                                psi = U*lmc.plus_n
                                
                                exp_val = psi.conj().T.dot(lmc.Ham).dot(psi)
                                exp_val = abs(exp_val.item())
                                print('ev={} (total loop time: {}s)'.format(exp_val, time.time() - t))
    
                                results.append((g1,b1, g2,b2,g3,b3,exp_val))
                                if exp_val > running_best[0]:
                                    running_best = (exp_val/n, g1, b1, g2, b2,g2,g3)
                                    print("ev={0}, {}}".format(*running_best))
                                
                                if c1 % 100 == 0:
                                    print('saving! {}/{}'.format(c1, total))
                                    with open(filename, 'a', newline='') as csvfile:
                                        writer = csv.writer(csvfile)
                                        for res in results:
                                            writer.writerow(res)

                c1 = c1 + 1



    def test_C_2_6(self):
        C2_5 = construct_ham_local2_as_arr(2, 5)
        I_by_C2_5 = np.kron(I, np.diag(C2_5))
        C2_6 = construct_ham_local2_as_arr(2, 6)
        self.assertTrue(np.allclose(I_by_C2_5, np.diag(C2_6)))

    def test_find_pairs(self):
        n = 7
        for i in range(n):
            C2_5 = construct_ham_local2_as_arr(i, n)
            I_by_C2_5 = np.kron(np.diag(C2_5), I)
            for j in range(n+1):
                # I_by_C2_5 = np.kron(I, np.diag(C2_5))
                C3_6 = construct_ham_local2_as_arr(j, n+1)
                if np.allclose(I_by_C2_5, np.diag(C3_6)):
                    print('{},{}'.format(i,j))

    def test_permuting_Cj(self):
        n = 9
        C0 = construct_ham_local2_as_arr(0, n)
        for j in range(1,n):
            Cj = construct_ham_local2_as_arr(j, n)

            for v in range(n):
                newv = shift(v,n-j,n)
                self.assertEqual(C0[newv], Cj[v], 'Error for newv={}, v={},j={}'.format(newv, v,j))

    # WE ONLY NEED C2_5!!!!
    def test_find_C2_n_using_C2_5(self):
        C2_5 = construct_ham_local2_as_arr(2, 5)
        for n in range(6,15):
            print('Getting away with n={}'.format(n))
            C2_n = construct_ham_local2_as_arr(2, n)
            for cut in range(2**n):
                cut_in_5 = project(cut, n)
                # print('cut={},cut_in_five={}'.format(cut,cut_in_5))
                self.assertEqual(C2_n[cut], C2_5[cut_in_5], 'C2_n[{0}]={1} should be equal to C2_5[{2}]={3}'.format(cut,C2_n[cut], cut_in_5, C2_5[cut_in_5]))

    def test_full_ham_using_C2_5(self):
        C2_5 = construct_ham_local2_as_arr(2, 5)
        for n in range(6,10): # can go to maybe 18
            print('Testing |V|={}'.format(n))
            H = ham_source_of_truth(n)
            for cut in range(2**n):
                actual = H[cut]
                cut_in_5 = project(cut, n)
                claim = C2_5[cut_in_5]
                for j in range(1,n):
                    cut_in_5 = project(shift(cut,j,n), n)
                    claim = claim + C2_5[cut_in_5]
                claim = claim / 3
                # print('<{0}|H_{2}|{0}> = {1}'.format(cut, claim,n))
                self.assertEqual(actual, claim)

    def test_ham_using_one_local_term(self):
        n = 6
        H = ham_source_of_truth(n)
        C2_6 = construct_ham_local2_as_arr(2, n)
        for cut in range(2**n):
            actual = H[cut]
            claim = C2_6[cut]
            for j in range(1,n):
                claim = claim + C2_6[shift(cut,j,n)]
            self.assertEqual(actual, claim/3)


    def test_hamiltonians(self):
        # C = construct_ham_local(0,4)
        C2 = ham_source_of_truth(9)
        C3 = construct_full_ham2(9).diagonal()
        self.assertTrue(np.allclose(C3,C2))




    def xtest_tensor_to_full_local_term(self):
        C2_5 = construct_ham_local2_as_arr(2, 5)
        for n in range(6,11):
            print('running against size {}'.format(n))
            C2_n = construct_ham_local2_as_arr(2, n)
            tensored = reduce(np.kron, [*[I for _ in range(n-5)], np.diag(C2_5)])
            self.assertTrue(np.allclose(tensored,np.diag(C2_n)), 'Failed for n={}'.format(n))
        

    def test_consistency(self):
        n = 7
        
        C = construct_ham_local(0,3)
        # C = ham_source_of_truth(n)

        gamma = 0.5975309227127786
        beta = 0.3144734246243383
        lmc = LocalMaxCut(C)
        exp_val = lmc.schrodinger_ev([expi(lmc.Ham,gamma)], [expi(lmc.X_sum,beta)])
        # psi = np.array(lmc.build_psi(gamma,beta))
        # exp_val2 = psi.conj().T.dot(C).dot(psi)
        self.assertTrue(np.allclose(exp_val/n, 0.939374559336945), 'Gotta beat 0.939374559336945')


    def test_shift(self):
        test = 9 # 01001
        shifted = shift(test,1, 5)
        expected = 18 # 10010
        self.assertEqual(shifted, expected)

    def test_printshifts(self):
        shifted = [shift(i,1,5)+1 for i in range(2**5)]
        print(shifted)

    
    def test_shrodinger_against_feynman(self):
        n = 13
        g = 0.5
        b = 0.75

        lmc = LocalMaxCut(np.diag(ham_source_of_truth(n)))

        t = time.time()
        exp_val = lmc.schrodinger_ev([construct_U_gamma(g, n)], [construct_U_gamma(b, n)])
        t1 = time.time() - t
        t = time.time()
        exp_val_ez2 = feynman_exp_val(n,g,b)
        t2 = time.time() - t
        print('schrodinger={} ({}s), feynman={} ({}s)'.format(norm(exp_val), t1, norm(exp_val_ez2), t2))

    def test_easy_U_beta(self):
        n = 7
        lmc = LocalMaxCut(np.diag(ham_source_of_truth(n)))
        B = X_sum(n)
        beta = 0.6

        U = expi(B, beta)
        for row in range(len(U)):
            for col in range(len(U[row])):
                self.assertTrue(np.allclose(U[row][col], lmc.U_beta_entry(beta, row, col)))

    def test_easy_U_gamma(self):
        n = 7
        C = np.diag(ham_source_of_truth(n))
        lmc = LocalMaxCut(C)
        gamma = 0.6

        U = expi(C, gamma)
        for row in range(len(U)):
            # for col in range(len(U[row])):
            v = lmc.U_gamma_entry(gamma, row)
            self.assertTrue(np.allclose(U[row][row], v))

    def test_easy_H_jj(self):
        for n in range(6, 10):
            C = np.diag(ham_source_of_truth(n))
            # lmc = LocalMaxCut(C)

            for cut in range(len(C)):
                v = H_jj_ring_fast(cut, n)
                self.assertTrue(np.allclose(C[cut][cut], v), 'Failure: n={}, cut={}'.format(n, cut))

    def xtest_compare_matrix_exponential(self):
        C = np.diag(ham_source_of_truth(5))
        lmc = LocalMaxCut(C)

        g1 = 1.23148465184464
        b1 = 0.149856354468
        g2 = 1.45437296484684
        b2 = 0.1461894648681
        U_gammas = [lmc.U_gamma_operator(g1), lmc.U_gamma_operator(g2)]
        U_betas = [lmc.U_beta_operator(b1), lmc.U_beta_operator(b2)]
        
        # U_gammas = [lmc.U_gamma_operator(g1)]
        # U_betas = [lmc.U_beta_operator(b1)]

        n = 3
        for _ in range(n):
            t = time.time()
            construct_unitary(U_gammas, U_betas)
            t1 = time.time() - t
            t = time.time()
            construct_unitary_gpu(U_betas, U_gammas)
            t2 = time.time() - t
            print('t1={}, t2={}'.format(t1,t2))



        U1 = construct_unitary(U_gammas, U_betas)
        U2 = construct_unitary_gpu(U_betas, U_gammas)
        self.assertTrue(np.allclose(U1,U2))

if __name__ == '__main__':
    unittest.main()
    