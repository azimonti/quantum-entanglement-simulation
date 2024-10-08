#!/usr/bin/env python3
'''
/************************/
/*  mod_spin_operators  */
/*      Version 1.0     */
/*      2024/05/11      */
/************************/
'''
import cmath
import math
import numpy as np
import random
import sys


class SingleSpin:

    def __init__(self, basis: str = 'ud'):
        self.__basis = basis
        self.__state = None
        if (basis == 'ud'):
            self.__u = np.array([[1], [0]], dtype=complex)
            self.__d = np.array([[0], [1]], dtype=complex)
            self.__r = np.array(
                [[1 / np.sqrt(2)], [1 / np.sqrt(2)]], dtype=complex)
            self.__l = np.array(
                [[1 / np.sqrt(2)], [-1 / np.sqrt(2)]], dtype=complex)
            self.__i = np.array(
                [[1 / np.sqrt(2)], [1j / np.sqrt(2)]], dtype=complex)
            self.__o = np.array(
                [[1 / np.sqrt(2)], [-1j / np.sqrt(2)]], dtype=complex)
            self.__sz = np.array([[1, 0], [0, -1]], dtype=complex)
            self.__sx = np.array([[0, 1], [1, 0]], dtype=complex)
            self.__sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        elif (basis == 'rl'):
            self.__u = np.array(
                [[1 / np.sqrt(2)], [1 / np.sqrt(2)]], dtype=complex)
            self.__d = np.array(
                [[1 / np.sqrt(2)], [-1 / np.sqrt(2)]], dtype=complex)
            self.__r = np.array([[1], [0]], dtype=complex)
            self.__l = np.array([[0], [1]], dtype=complex)
            self.__i = np.array(
                [[(1 + 1j) / 2], [(1 - 1j) / 2]], dtype=complex)
            self.__o = np.array(
                [[(1 - 1j) / 2], [(1 + 1j) / 2]], dtype=complex)
        elif (basis == 'io'):
            self.__u = np.array(
                [[1 / np.sqrt(2)], [1 / np.sqrt(2)]], dtype=complex)
            self.__d = np.array(
                [[1j / np.sqrt(2)], [-1j / np.sqrt(2)]], dtype=complex)
            self.__r = np.array(
                [[(1 + 1j) / 2], [(1 - 1j) / 2]], dtype=complex)
            self.__l = np.array(
                [[(1 - 1j) / 2], [(1 + 1j) / 2]], dtype=complex)
            self.__o = np.array([[1], [0]], dtype=complex)
            self.__i = np.array([[0], [1]], dtype=complex)
        else:
            raise NotImplementedError(
                "Basis " + basis + "not Implemented")

    @property
    def u(self):
        return self.__u

    @property
    def d(self):
        return self.__d

    @property
    def r(self):
        return self.__r

    @property
    def l(self):
        return self.__l

    @property
    def i(self):
        return self.__i

    @property
    def o(self):
        return self.__o

    @property
    def s_z(self):
        if (self.__basis != 'ud'):
            raise NotImplementedError(
                "S_z for basis " + self.__basis + "not Implemented")
        return self.__sz

    @property
    def s_x(self):
        if (self.__basis != 'ud'):
            raise NotImplementedError(
                "S_x for basis " + self.__basis + "not Implemented")
        return self.__sx

    @property
    def s_y(self):
        if (self.__basis != 'ud'):
            raise NotImplementedError(
                "S_y for basis " + self.__basis + "not Implemented")
        return self.__sy

    @property
    def psi(self):
        return self.__state

    @psi.setter
    def psi(self, value):
        # check that length is unitary
        assert math.isclose(np.linalg.norm(value), 1)
        self.__state = value

    def theta(self):
        assert self.__state
        return 2 * np.arccos(np.linalg.norm(self.__state[0][0]))

    def phi(self):
        assert self.__state
        if self.__state[1][0] != 0:
            return cmath.phase(self.__state[1][0])
        else:
            return 0

    def angles(self):
        return [self.theta(), self.phi()]


class TwoSpin:

    def __init__(self, basis: str = 'ud'):
        self.__basis = basis
        self.__state = None
        if (basis == 'ud'):
            u = np.array([[1], [0]], dtype=complex)
            d = np.array([[0], [1]], dtype=complex)
            self.__b = [
                np.kron(u, u), np.kron(u, d),
                np.kron(d, u), np.kron(d, d)
            ]
            self.__bmap = {'uu': 0, 'ud': 1, 'du': 2, 'dd': 3}
            self.__s = [
                np.array([[1, 0], [0, -1]], dtype=complex),
                np.array([[0, 1], [1, 0]], dtype=complex),
                np.array([[0, -1j], [1j, 0]], dtype=complex),
                np.array([[1, 0], [0, 1]], dtype=complex)
            ]
            self.__smap = {'z': 0, 'x': 1, 'y': 2, 'I': 3}
        else:
            raise NotImplementedError(
                "Basis " + basis + "not Implemented")

    @property
    def psi(self):
        return self.__state

    @ psi.setter
    def psi(self, value):
        # check that length is unitary
        assert math.isclose(np.linalg.norm(value), 1)
        self.__state = value

    def BasisVector(self, s):
        return self.__b[self.__bmap[s]]

    def Sigma(self, sA: str, sB: str):
        return np.kron(
            self.__s[self.__smap[sA]], self.__s[self.__smap[sB]])

    def Sigma_A(self, s: str):
        return np.kron(self.__s[self.__smap[s]], self.__s[3])

    def Sigma_B(self, s: str):
        return np.kron(self.__s[3], self.__s[self.__smap[s]])

    def Expectation(self, sA: str, sB: str):
        if (sA == 'I'):
            # expectation of  System B
            return np.linalg.multi_dot([
                self.__state.conj().T, self.Sigma_B(sB), self.__state])[0, 0]
        elif (sB == 'I'):
            # expectation of  System A
            return np.linalg.multi_dot([
                self.__state.conj().T, self.Sigma_A(sA), self.__state])[0, 0]
        else:
            # expectation of the composite system
            return np.linalg.multi_dot([
                self.__state.conj().T, self.Sigma(sA, sB), self.__state])[0, 0]

    def ProductState(self, A: np.array, B: np.array):
        # check that length is unitary
        assert math.isclose(np.linalg.norm(A), 1)
        assert math.isclose(np.linalg.norm(B), 1)
        self.psi = np.kron(A, B)

    def Singlet(self):
        self.psi = 1 / np.sqrt(2) * (self.__b[1] - self.__b[2])

    def Triplet(self, i: int):
        match i:
            case 1:
                self.psi = 1 / np.sqrt(2) * (self.__b[1] + self.__b[2])
            case 2:
                self.psi = 1 / np.sqrt(2) * (self.__b[0] + self.__b[3])
            case 3:
                self.psi = 1 / np.sqrt(2) * (self.__b[0] - self.__b[3])
            case _:
                raise ValueError("Incorrect index " + str(i))

    def Measure(self, directions1: np.ndarray,
                directions2: np.ndarray, simulate_1: bool = True,
                update: bool = False):
        '''
        Perform the measurement of the spin of two directions 1 and 2,
        which one to simulate is decided by the caller.
        '''
        def Rho1(psi: np.ndarray):
            return np.outer(psi, psi.conj()).reshape(
                (2, 2, 2, 2)).trace(axis1=1, axis2=3)

        def Rho2(psi: np.ndarray):
            return np.outer(psi, psi.conj()).reshape((
                2, 2, 2, 2)).trace(axis1=0, axis2=2)

        psi = self.__state
        # Define the projector operator for the "+1" state
        direction1_p1 = np.array(directions1[0])
        direction1_m1 = np.array(directions1[1])
        direction2_p1 = np.array(directions2[0])
        direction2_m1 = np.array(directions2[1])
        projector1_p1 = np.outer(direction1_p1, direction1_p1.conj())
        projector1_m1 = np.outer(direction1_m1, direction1_m1.conj())
        projector2_p1 = np.outer(direction2_p1, direction2_p1.conj())
        projector2_m1 = np.outer(direction2_m1, direction2_m1.conj())

        if simulate_1:
            # Create 4x4 projectors for the two-spin system
            projector_p1_s = np.kron(projector1_p1, np.eye(2))
            projector_m1_s = np.kron(projector1_m1, np.eye(2))
            projector_p11 = projector1_p1
            projector_p21 = projector2_p1

            # Calculate the reduced density matrix for the first spin
            # which is measured
            rho_i = Rho1(psi)
        else:
            projector_p1_s = np.kron(np.eye(2), projector2_p1)
            projector_m1_s = np.kron(np.eye(2), projector2_m1)
            projector_p11 = projector2_p1
            projector_p21 = projector1_p1
            rho_i = Rho2(psi)

        # Calculate the probability of this first spin being "+1"
        prob_p11 = np.linalg.norm(np.trace(np.dot(projector_p11, rho_i)))
        # Generate a random number between 0 and 1
        random_number1 = random.uniform(0, 1)

        # Perform the measurement in apparatus direction
        sp1 = 1 if random_number1 < prob_p11 else -1

        # reduce psi projecting on the direction
        psi_r = np.dot(projector_p1_s, psi) if sp1 == 1 else \
            np.dot(projector_m1_s, psi)
        # Normalize psi_r
        psi_r = psi_r / np.linalg.norm(psi_r)
        # Calculate the reduced density matrix for the second spin
        # with the collapsed wave function for the first system
        rho_j = Rho2(psi_r) if simulate_1 else Rho1(psi_r)

        # Calculate the probability of "system 2" being "+1"
        prob_p12 = np.linalg.norm(np.trace(np.dot(projector_p21, rho_j)))

        # Generate a random number between 0 and 1
        random_number2 = random.uniform(0, 1)

        sp2 = 1 if random_number2 < prob_p12 else -1
        if update:
            self.__state = psi
        return (sp1, sp2)


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise RuntimeError('Must be using Python 3')
    pass
