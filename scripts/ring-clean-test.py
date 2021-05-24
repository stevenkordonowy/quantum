import unittest
# import sys
from ringclean import * 
from LMC import * 
from numpy.linalg import norm
import numpy as np
import math
import concurrent.futures
import time
import csv
import time
from qutip import *
import multiprocessing.dummy as mp 
from joblib import Parallel, delayed
import random
import datetime
        
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

def stringify_list(l):
    return ','.join(str(it) for it in l)

def test_print(s):
    t = datetime.datetime.fromtimestamp(time.time())
    print('{}: {}'.format(t, s))

class TestRingClean(unittest.TestCase):

    def setUp(self):
        test_print("\n########################\nRunning method:{}\n########################".format(self._testMethodName))

    def xtest_hamming(self):
        s = ''
        for i in range(11):
            for j in range(i, 11):
                s += '{{ {}, {} }} -> {}, '.format(i,j,hamming_dist(i,j))
        print(s)

        s = ''
        # for i in range()

    def test_same_sizes(self):
        self.assertRaises(ValueError, AlgInstance, 5, [1], [1,2])
        self.assertRaises(ValueError, AlgInstance, 5, [1, 2], [1])
        
    def test_cannot_be_zero(self):
        self.assertRaises(ValueError, AlgInstance, 5, [], [])

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

        alg = AlgInstance(n, [0], [0])
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

        test_print('About to instantiate alg')
        alg = AlgInstance(n, [gamma], [beta])

        test_print('Run outer loop:')
        for x in range(2 ** n):
            # test_print('checking outer={}'.format(x))
            # Parallel(n_jobs=4, prefer="threads")(delayed(self.verify_inner)(alg, x, y, Ug, Ub) for y in range(2 ** n))
            for y in range(2 ** n):
                self.verify_inner(alg, x, y, Ug, Ub)
                # U_el = alg.U_el(x,y,1)
                # expe = Ug[x][x] * Ub[x][y]
                # self.assertTrue(np.allclose(U_el, expe), '{} should be equal to {}'.format(U_el, expe))

    def verify_inner(self, alg, x, y, Ug, Ub):
        U_el = alg.U_el(x,y,1)
        expe = Ug[x][x] * Ub[x][y]
        self.assertTrue(np.allclose(U_el, expe), '{} should be equal to {}'.format(U_el, expe))

    def test_easy_p1(self):
        n = 7
        g = 0.6
        b = 0.6

        alg = AlgInstance(n, [g], [b])
        ev = alg.run_vec()
        test_print(ev)
        self.assertTrue(np.allclose(ev, 5.45133990578963), '{} should be equal to {}'.format(ev, 5.45133990578963))


    def test_build_states_ez(self):
        n = 7
        # g = 0.5711986642890533
        # b = 0.3173325912716963
        g = 0
        b = 0
        alg = AlgInstance(n, [g], [b])

        expected = alg.build_state()
        newer = alg.build_state_in_half()
        for j in range(int(len(expected) / 2)):
            self.assertTrue(np.allclose(expected[j], expected[len(expected) - j - 1]))
            self.assertTrue(np.allclose(expected[j], newer[j]), 'expected[{}] ={}, newer[{}] = {}'.format(j, expected[j],j, newer[j]))


    def test_build_states(self):
        n = 7
        g = 0.5711986642890533
        b = 0.3173325912716963
        # g = 0
        # b = 0
        alg = AlgInstance(n, [g], [b])

        expected = alg.build_state()
        newer = alg.build_state_in_half()
        for j in range(int(len(expected) / 2)):
            self.assertTrue(np.allclose(expected[j], expected[len(expected) - j - 1]))
            self.assertTrue(np.allclose(expected[j], newer[j]), 'expected[{}] ={}, newer[{}] = {}'.format(j, expected[j],j, newer[j]))
            

    def test_easy_p1_vec(self):
        n = 7
        g = 0.5711986642890533
        b = 0.3173325912716963
        # 0.5711986642890533,0.3173325912716963,6.573418608599479

        # g = 0
        # b = 0
        exp_ev = 6.573418608599479

        alg = AlgInstance(n, [g], [b])
        t = time.time()
        ev = alg.run_vec()
        t1 = time.time() - t
        
        t = time.time()
        ev2 = alg.run_vec_half()
        t2 = time.time() - t
        test_print('t1={}, t2={}'.format(t1, t2))
        self.assertTrue(np.allclose(ev, exp_ev), '{} should be equal to {}'.format(ev, exp_ev))


        self.assertTrue(np.allclose(ev, ev2), '{} should be equal to {}'.format(ev, ev2))


    def xtest_easy_p2_vec(self):
        n = 11
        # New best! ev=10.726089729072456 (0.9750990662793142%), g=0.4268436281766437,1.080530884529931, b=0.4046214059544214,0.17615535148860537
        g1 = 0.4268436281766437
        g2 = 1.080530884529931
        b1 = 0.4046214059544214
        b2 = 0.17615535148860537
        gamma = [g1,g2]
        beta = [b1, b2]

        alg = AlgInstance(n, gamma, beta)
        exp_ev = 10.726089729072456

        test_print('Kicking off run_vec_half')
        t = time.time()
        ev2 = alg.run_vec_half()
        t1 = time.time() - t
        test_print('first: {}'.format(t1))

        t = time.time()
        ev = alg.run_vec()
        t2 = time.time() - t
        test_print('First: {}s, second: {}s'.format(t1, t2))
        # self.assertTrue(np.allclose(ev, 6.454958892059133), '{} should be equal to {}'.format(ev, 5.45133990578963))
        self.assertTrue(np.allclose(ev, ev2, exp_ev), '{} should be equal to {}'.format(ev, ev2))
        
    def test_p1_full(self):
        save = True

        n = 7
        num = 250

        test_print('#### Running QAOA for p=1, d=1 on size n={}'.format(n))
        best = -1

        if save:
            filename = 'n{}p1.csv'.format(n)
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['g1', 'b', 'exp_val'])

        c = 1
        results = []
        t = time.time()
        for g in linspace(0.0, pi, num = num):
            for b in linspace(0.0, pi/2, num = num):
                alg = AlgInstance(n, [g], [b])

                ev = alg.run_vec_half()
                # ev2 = alg.run_p1_vec()
                # self.assertTrue(np.allclose(ev, ev2), '{} should equal {} for angles {}, {}'.format(ev, ev2,g,b))

                results.append((g, b, ev))

                if ev > best:
                    best = ev
                    test_print('New best! ev={} ({}%), g={}, b={}'.format(best, best / n, g, b))

                if c % 100 == 0:
                    t2 = time.time() - t
                    test_print('last ev={}, avg time = {}s'.format(ev, t2 / c))   

                    # Safety check
                    # slow_ev = ev = alg.run_vec()
                    # self.assertTrue(np.allclose(ev, slow_ev), '{} should equal {} for angles {}, {}'.format(ev, slow_ev, g, b))

                    if save:
                        test_print('saving! {}/{}'.format(c, num * num))
                        with open(filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            for res in results:
                                writer.writerow(res)

                    results = []
                
                c = c + 1

        test_print('Done! {}/{}'.format(c, num * num))
        if save:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for res in results:
                    writer.writerow(res)

        test_print('Took {}s'.format(time.time() - t))
        test_print('Overall best={} ({}%)'.format(best, best / n))

    def xtest_p2_using_p1s_best(self):
        save = True

        n = 11
        num = 25
        saving_num = num
        total = num ** 2

        test_print('#### Running QAOA for p=2, d=1 on size n={}'.format(n))
        best = -1
        # "[0.36510401109286783, 0.9170054232099937]","[0.4121807298358501, 0.14528686746331176]",10.649077211393838
        p1_g = 0.6029319234162229
        p1_b = 0.3173325912716963

        if save:
            filename = 'n{}p2_usingp1.csv'.format(n)
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['g1', 'g2', 'b1', 'b2', 'exp_val'])

        c = 1
        results = []
        t = time.time()
        for g1 in linspace(0.01, 2*pi, num = num):
            for b1 in linspace(0.01, pi, num = num):
                # g1 + g2 = g0
                g = [g1, p1_g - g1]
                b = [b1, p1_b - b1]
                alg = AlgInstance(n, g, b)

                ev = alg.run_vec_half()

                results.append((stringify_list(g), stringify_list(b), ev))

                if ev > best:
                    best = ev
                    test_print('New best! ev={} ({}%), g={}, b={}'.format(best, best / n, stringify_list(g), stringify_list(b)))

                # g1 * g2 = g0
                g = [g1, p1_g / g1]
                b = [b1, p1_b / b1]
                alg = AlgInstance(n, g, b)

                ev = alg.run_vec_half()

                results.append((stringify_list(g), stringify_list(b), ev))

                if ev > best:
                    best = ev
                    test_print('New best! ev={} ({}%), g={}, b={}'.format(best, best / n, stringify_list(g), stringify_list(b)))

                if c % saving_num == 0:
                    t2 = time.time() - t
                    test_print('last ev={}, avg time = {}s'.format(ev, 2 * t2 / c))

                    # Safety check
                    # slow_ev = ev = alg.run_vec()
                    # self.assertTrue(np.allclose(ev, slow_ev), '{} should equal {} for angles {}, {}'.format(ev, slow_ev, g, b))

                    if save:
                        test_print('saving! {}/{}'.format(c, total))
                        with open(filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            for res in results:
                                writer.writerow(res)

                    results = []
                
                c = c + 1

        if save:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for res in results:
                    writer.writerow(res)

        test_print('Took {}s'.format(time.time() - t))
        test_print('Overall best={} ({}%)'.format(best, best / n))        

    def xtest_p2_full(self):
        save = True

        n = 11
        num = 10
        saving_num = num ** 2
        total = num ** 4

        test_print('#### Running QAOA for p=2, d=1 on size n={}'.format(n))
        best = -1

        if save:
            filename = 'n{}p2.csv'.format(n)
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['g1', 'g2', 'b1', 'b2', 'exp_val'])

        c = 1
        results = []
        t = time.time()
        for g1 in linspace(0.1, pi/2, num = num):
            for b1 in linspace(0.1, pi/4, num = num):
                for g2 in linspace(0.1, pi/2, num = num):
                    for b2 in linspace(0.1, pi/4, num = num):
                        g = [g1, g2]
                        b = [b1, b2]
                        alg = AlgInstance(n, g, b)

                        ev = alg.run_vec_half()

                        results.append((g, b, ev))

                        if ev > best:
                            best = ev
                            test_print('New best! ev={} ({}%), g={}, b={}'.format(best, best / n, stringify_list(g), stringify_list(b)))

                        if c % saving_num == 0:
                            t2 = time.time() - t
                            test_print('last ev={}, avg time = {}s'.format(ev, t2 / c))

                            # Safety check
                            # slow_ev = ev = alg.run_vec()
                            # self.assertTrue(np.allclose(ev, slow_ev), '{} should equal {} for angles {}, {}'.format(ev, slow_ev, g, b))

                            if save:
                                test_print('saving! {}/{}'.format(c, total))
                                with open(filename, 'a', newline='') as csvfile:
                                    writer = csv.writer(csvfile)
                                    for res in results:
                                        writer.writerow((stringify_list(res[0]), stringify_list(res[1]), res[2]))

                            results = []
                        
                        c = c + 1

        if save:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for res in results:
                    writer.writerow((stringify_list(res[0]), stringify_list(res[1]), res[2]))

        test_print('Took {}s'.format(time.time() - t))
        test_print('Overall best={} ({}%)'.format(best, best / n))

    def xtest_p2_random(self):
        save = True

        n = 11
        num = 1000
        when_to_save = math.floor(math.sqrt(num) / 3)

        test_print('#### Running QAOA for p=2, d=1 on size n={}'.format(n))
        best = -1

        if save:
            filename = 'p2-{}-random.csv'.format(n)
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['g1', 'g2', 'b1', 'b2', 'exp_val'])

        results = []
        gammas = list(linspace(pi/20, pi/2, num = 1000))
        betas = list(linspace(pi/50, pi/4, num = 1000))
        c = 1
        t = time.time()
        for _ in range(num):
            g1, g2 = random.sample(gammas, 2)
            b1, b2 = random.sample(betas, 2)
            g = [g1, g2]
            b = [b1, b2]
            alg = AlgInstance(n, g, b)
            test_print('Running with angles g={}, b={}'.format(stringify_list(g), stringify_list(b)))

            # ev = alg.run_p1()
            ev = alg.run_vec_half()
            # self.assertTrue(np.allclose(ev, ev2), '{} should equal {} for angles {}, {}'.format(ev, ev2,g1,b1))

            results.append((g, b, ev))

            if ev > best:
                best = ev
                test_print('New best! ev={} ({}%), g={}, b={}'.format(best, best / n, stringify_list(g), stringify_list(b)))

            if c % when_to_save == 0:
                t2 = time.time() - t
                test_print('last ev={}, curr_best={}, avg time = {}s'.format(ev, best, t2 / c))
                if save:
                    test_print('saving! {}/{}'.format(c, num))
                    with open(filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for res in results:
                            writer.writerow(res)

                results = []

            c += 1

        if save:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for res in results:
                    writer.writerow(res)

        test_print('Took {}s'.format(time.time() - t))
        test_print('Overall best={} ({}%)'.format(best, best / n))

    def xtest_p2_around_p2best(self):
        save = True

        n = 11
        num = 25
        saving_num = num
        total = num ** 2

        test_print('#### Running QAOA for p=2, d=1 on size n={}'.format(n))
        best = -1

        # ev=10.72777002875542 (0.9752518207959473%), g=0.4268436281766437,1.0491698416354398, b=0.4046214059544214,0.17472877341273718
        g1_best = 0.4268436281766437
        g2_best = 1.0491698416354398
        b1_best = 0.4046214059544214
        b2_best = 0.17472877341273718
        radius = 0.001

        if save:
            filename = 'n{}p2_usingp2.csv'.format(n)
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['g1', 'g2', 'b1', 'b2', 'exp_val'])

        c = 1
        results = []
        t = time.time()
        for g2 in linspace((1-radius) * g2_best, (1+radius) * g2_best, num = num):
            for b2 in linspace((1-radius) * b2_best, (1+radius) *b2_best, num = num):
                g = [g1_best, g2]
                b = [b1_best, b2]
                alg = AlgInstance(n, g, b)

                ev = alg.run_vec_half()

                results.append((g, b, ev))

                if ev > best:
                    best = ev
                    test_print('New best! ev={} ({}%), g={}, b={}'.format(best, best / n, stringify_list(g), stringify_list(b)))

                if c % saving_num == 0:
                    t2 = time.time() - t
                    test_print('last ev={}, avg time = {}s'.format(ev, 2 * t2 / c))

                    # Safety check
                    # slow_ev = ev = alg.run_vec()
                    # self.assertTrue(np.allclose(ev, slow_ev), '{} should equal {} for angles {}, {}'.format(ev, slow_ev, g, b))

                    if save:
                        test_print('saving! {}/{}'.format(c, total))
                        with open(filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            for res in results:
                                writer.writerow((stringify_list(res[0]), stringify_list(res[1]), res[2]))

                    results = []
                
                c = c + 1

        for g1 in linspace((1-radius) * g1_best, (1+radius) * g1_best, num = num):
            for b1 in linspace((1-radius) * b1_best, (1+radius) *b1_best, num = num):
                g = [g2_best, g1]
                b = [b2_best, b1]
                alg = AlgInstance(n, g, b)

                ev = alg.run_vec_half()

                results.append((g, b, ev))

                if ev > best:
                    best = ev
                    test_print('New best! ev={} ({}%), g={}, b={}'.format(best, best / n, stringify_list(g), stringify_list(b)))

                if c % saving_num == 0:
                    t2 = time.time() - t
                    test_print('last ev={}, avg time = {}s'.format(ev, 2 * t2 / c))


                    if save:
                        test_print('saving! {}/{}'.format(c, total))
                        with open(filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            for res in results:
                                writer.writerow((stringify_list(res[0]), stringify_list(res[1]), res[2]))

                    results = []
                
                c = c + 1

        if save:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for res in results:
                    writer.writerow(res)

        test_print('Took {}s'.format(time.time() - t))
        test_print('Overall best={} ({}%)'.format(best, best / n))        

    def xtest_easy_p3_vec(self):
        n = 15
        g1 = 0.6
        g2 = 1.1
        g3 = 0.8
        b1 = 0.6
        b2 = 0.3
        b3 = .9

        gamma = [g1,g2,g3]
        beta = [b1, b2, b3]

        test_print('About to instantiate alg')
        alg = AlgInstance(n, gamma, beta)

        test_print('Kicking off p=3:')        
        t = time.time()
        ev = alg.run_vec_half()
        t1 = time.time() - t
        test_print('First: {}s, ev={}'.format(t1, ev))
        # t = time.time()
        # ev2 = alg.run_vec()
        # t2 = time.time() - t
        # test_print('First: {}s, second: {}'.format(t1, t2))

    def test_f2(self):
        # print(nbhd_2_boolean())
        # for a in [-1, 1]:
        #     for b in [-1, 1]:
        #         for c in [-1, 1]:
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    val = f_general(a,[b,c])
                    print('{}, {}, {}: {}'.format(a,b,c, val))
                    if val[0]:
                        self.assertEqual(1, val[1], 'Should be true: {},{},{}'.format(a,b,c))
                    else:
                        self.assertEqual(0, val[1], 'Should be false: {},{},{}'.format(a,b,c))

    def test_f3(self):
        
        for a in [-1, 1]:
            for b in [-1, 1]:
                for c in [-1, 1]:
                    for d in [-1, 1]:
        # for a in [0, 1]:
        #     for b in [0, 1]:
        #         for c in [0, 1]:
        #             for d in [0, 1]:
                        val = f3(a,b,c,d)
                        print('{}, {}, {}, {}: {}'.format(a,b,c,d, val))
                        if val[0]:
                            self.assertEqual(1, val[1], 'Should be true')
                        else:
                            self.assertEqual(0, val[1], 'Should be false')

    def test_hypercube(Self):
        construct_H_cube()

    def xtest_f4(self):
        # for a in [-1, 1]:
        #     for b in [-1, 1]:
        #         for c in [-1, 1]:
        #             for d in [-1, 1]:
        #                 for e in [-1, 1]:
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for d in [0, 1]:
                        for e in [0, 1]:
                            val = f_general(a,[b,c,d,e])
                            print('{}, {}, {}, {}, {}: {}'.format(a,b,c,d,e, val))
                            if val[0]:
                                self.assertEqual(1, val[1], 'Should be true')
                            else:
                                self.assertEqual(0, val[1], 'Should be false')
    def xtest_f5(self):
        # for a in [-1, 1]:
        #     for b in [-1, 1]:
        #         for c in [-1, 1]:
        #             for d in [-1, 1]:
        #                 for e in [-1, 1]:
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for d in [0, 1]:
                        for e in [0, 1]:
                            for f in [0,1]:
                                val = f_general(a,[b,c,d,e,f])
                                print('{}, {}, {}, {}, {}, {}: {}'.format(a,b,c,d,e,f, val))
                                if val[0]:
                                    self.assertEqual(1, val[1], 'Should be true')
                                else:
                                    self.assertEqual(0, val[1], 'Should be false')

    def xtest_f6(self):
        # for a in [-1, 1]:
        #     for b in [-1, 1]:
        #         for c in [-1, 1]:
        #             for d in [-1, 1]:
        #                 for e in [-1, 1]:
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for d in [0, 1]:
                        for e in [0, 1]:
                            for f in [0,1]:
                                for g in [0,1]:
                                    val = f_general(a,[b,c,d,e,f,g])
                                    print('{}, {}, {}, {}, {}, {}, {}: {}'.format(a,b,c,d,e,f,g, val))
                                    if val[0]:
                                        self.assertEqual(1, val[1], 'Should be true')
                                    else:
                                        self.assertEqual(0, val[1], 'Should be false')


    def xtest_f7(self):
        # for a in [-1, 1]:
        #     for b in [-1, 1]:
        #         for c in [-1, 1]:
        #             for d in [-1, 1]:
        #                 for e in [-1, 1]:
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for d in [0, 1]:
                        for e in [0, 1]:
                            for f in [0,1]:
                                for g in [0,1]:
                                    for h in [0,1]:
                                        val = f_general(a,[b,c,d,e,f,g,h])
                                        # print('{}, {}, {}, {}, {}, {}, {}, {}: {}'.format(a,b,c,d,e,f,g,h, val))
                                        # if val[0]:
                                        #     self.assertEqual(1, val[1], 'Should be true')
                                        # else:
                                        #     self.assertEqual(0, val[1], 'Should be false')
    def xtest_print_bool_nbhds(self):
        print('d=2: {}'.format(nbhd_2_boolean()))
        print('\nd=3: {}'.format(nbhd_3_boolean()))
        print('\nd=4: {}'.format(nbhd_4_boolean()))
        print('\nd=5: {}'.format(nbhd_5_boolean()))
        print('\nd=6: {}'.format(nbhd_6_boolean()))
        print('\nd=7: {}'.format(nbhd_7_boolean()))

if __name__ == '__main__':
    unittest.main()
    
