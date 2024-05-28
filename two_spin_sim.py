#!/usr/bin/env python3
'''
/************************/
/*    two_spin_sim.py   */
/*    Version 1.0       */
/*      2024/05/11      */
/************************/
'''
import argparse
import math
from mod_spin_operators import SingleSpin, TwoSpin
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QPushButton, QSlider, QLabel
from PyQt6.QtWidgets import QVBoxLayout, QGridLayout
from PyQt6.QtWidgets import QWidget
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QFont, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect
from OpenGL.GL import (
    glClear, glClearColor, glEnable, glPushMatrix, glPopMatrix, glRotatef,
    glTranslatef, glBegin, glEnd, glVertex3f, glViewport, glMatrixMode,
    glLoadIdentity, glColor3f)
from OpenGL.GLU import gluPerspective, gluLookAt
from OpenGL.GL import (
    GL_DEPTH_TEST, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_QUADS, GL_LINES, glFlush, GL_PROJECTION, GL_MODELVIEW)
import random
import sys
import time
from types import SimpleNamespace

cfg = SimpleNamespace(
    stype=3, m=False, color_left=(0, 1, 0), color_right=(1, 0, 0),
    # additional coefficients  to to convert the real-space angles
    # into the corresponding angles in the Hilbert space (Bloch sphere)
    bloch_t=1.0, bloch_p=1.0,
    verbose=False)

description = (
    'This script simulates two spin following '
    'quantum mechanics principles.\n\n'
    'Simulation types available (-t SIMUL_TYPE, --simul_type SIMUL_TYPE):\n'
    '0 - No Time evolution\n'
    '1 - Product state\n'
    '    A = | u >\n'
    '    B = | d >\n'
    '2 - Product state\n'
    '    A = 1 / sqrt(2) * (| u > + | d >)\n'
    '    B = | u > / 2 + sqrt(3) / 2 * | d >\n'
    '3 - Singlet state\n'
    '    | Psi > = 1 / sqrt(2) * (| ud > - | du >) [DEFAULT]\n'
    '4 - Triplet state I\n'
    '    | Psi > = 1 / sqrt(2) * (| ud > + | du >)\n'
    '5 - Triplet state II\n'
    '    | Psi > = 1 / sqrt(2) * (| uu > + | dd >)\n'
    '6 - Triplet state III\n'
    '    | Psi > = 1 / sqrt(2) * (| uu > - | dd >)\n'
    '7 - Partially entangled state\n'
    '    | Psi > = sqrt(0.6) * | ud > - sqrt(0.4) * | du >\n'
    '    It requires a few hundred measurements to show the '
    'correct correlation σ^z = 1.00, σ^x = σ^y = 0.98\n\n'
    'The program measures each component separately when any of the '
    'two measure buttons is pressed and resets the status.\n'
    'It is possible to measure both systems at the same time when one of '
    'the measure buttons is pressed and the correlation between these '
    'measurements is shown setting the command line option '
    '"-m, --measure_both" (default = False).\n\n'
    'It is possible to set the color for the left and right apparatus '
    'with the command line option "-l, --color_left" (default = green) '
    'and "-r, --color_right (default = red).'
)


class SimulationThread(QThread):
    # Currently is not used but it is ready in case there is a time
    # evolution for example updating with a magnetic field applied
    result = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.current_state = None

    def run(self):
        spin = TwoSpin()
        s = SingleSpin()
        # initial condition for the case needed
        match cfg.stype:
            case 1:
                A = s.u
                B = s.d
                spin.ProductState(A, B)
            case 2:
                A = 1 / np.sqrt(2) * (s.u + s.d)
                B = s.u / 2 + np.sqrt(3) / 2 * s.d
                spin.ProductState(A, B)
            case 3:
                spin.Singlet()
            case 4:
                spin.Triplet(1)
            case 5:
                spin.Triplet(2)
            case 6:
                spin.Triplet(3)
            case 7:
                spin.psi = np.sqrt(0.6) * spin.BasisVector('ud') - \
                    np.sqrt(0.4) * spin.BasisVector('du')
            case _:
                raise ValueError(
                    f"Incorrect simulation type {cfg.stype}")
        while True:
            self.current_state = spin.psi
            self.result.emit(self.current_state)
            time.sleep(1)  # slows down the loop for demonstration purposes


class OpenGLWidget(QOpenGLWidget):

    def __init__(self, parent):
        super(OpenGLWidget, self).__init__(parent)
        self.a_thetaA = 0
        self.a_phiA = 0
        self.measurementA = None
        self.count_p1A = 0
        self.count_m1A = 0
        self.a_thetaB = 0
        self.a_phiB = 0
        self.measurementB = None
        self.count_p1B = 0
        self.count_m1B = 0
        self.current_state = None
        self.sigma = {'A': {'x': [], 'y': [], 'z': [], 'th_ph': []},
                      'B': {'x': [], 'y': [], 'z': [], 'th_ph': []}
                      }
        self.directions = {
            'z': np.array([[1, 0],
                           [0, 1]]),
            'x': np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                           [1 / np.sqrt(2), -1 / np.sqrt(2)]]),
            'y': np.array([[1 / np.sqrt(2), 1j / np.sqrt(2)],
                           [1 / np.sqrt(2), -1j / np.sqrt(2)]])
        }

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # Adjust the camera view
        gluLookAt(0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        glTranslatef(-3.0, 0.0, 0.0)  # Move to the left
        glPushMatrix()
        glColor3f(*cfg.color_left)
        glMatrixMode(GL_MODELVIEW)
        # Rotate the rectangle along Y-axis
        a_theta_deg = math.degrees(self.a_thetaA)
        a_phi_deg = math.degrees(self.a_phiA)
        # Apply the rotation
        glRotatef(-a_phi_deg, 0.0, 0.0, 1.0)
        glRotatef(a_theta_deg, 0.0, 1.0, 0.0)
        self.drawRectangleAndArrow()
        glPopMatrix()
        glTranslatef(6.0, 0.0, 0.0)  # Move to the right
        glPushMatrix()
        glColor3f(*cfg.color_right)
        glMatrixMode(GL_MODELVIEW)
        # Rotate the rectangle along Y-axis
        a_theta_deg = math.degrees(self.a_thetaB)
        a_phi_deg = math.degrees(self.a_phiB)
        # Apply the rotation
        glRotatef(-a_phi_deg, 0.0, 0.0, 1.0)
        glRotatef(a_theta_deg, 0.0, 1.0, 0.0)
        self.drawRectangleAndArrow()
        glPopMatrix()

        if len(self.sigma['A']['th_ph']) > 0:
            saz = np.mean(self.sigma['A']['z'])
            sax = np.mean(self.sigma['A']['x'])
            say = np.mean(self.sigma['A']['y'])
            sai = saz**2 + sax**2 + say**2
            painter = QPainter(self)
            painter.setFont(QFont('Arial', 14))
            painter.setPen(QColor(255, 255, 255))
            half_width = int(self.width() / 2)
            y = int(0.25 * self.height() + 55)
            rect = QRect(0, 0, half_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Σ< σ_Ai > = {sai:.2f}")
            y = int(0.25 * self.height() + 90)
            rect = QRect(0, 0, half_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< σ^Ay > = {say:.2f}")
            y = int(0.25 * self.height() + 125)
            rect = QRect(0, 0, half_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< σ^Ax > = {sax:.2f}")
            y = int(0.25 * self.height() + 160)
            rect = QRect(0, 0, half_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< σ^Az > = {saz:.2f}")
            y = int(0.25 * self.height() + 55)
            rect = QRect(0, y, half_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Measurement: {self.measurementA}")
            y = int(0.25 * self.height() + 90)
            rect = QRect(0, y, half_width, self.height() - y)
            painter.drawText(
                rect, Qt.AlignmentFlag.AlignCenter,
                f"Total Measurements: {len(self.sigma['A']['th_ph'])}")
            y = int(0.25 * self.height() + 125)
            rect = QRect(0, y, half_width, self.height() - y)
            prob_p1 = self.count_p1A / len(self.sigma['A']['th_ph']) * 100
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f" < +1 > = {prob_p1:.1f}%")
            y = int(0.25 * self.height() + 160)
            rect = QRect(0, y, half_width, self.height() - y)
            prob_m1 = self.count_m1A / len(self.sigma['A']['th_ph']) * 100
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< -1 > = {prob_m1:.1f}%")
            painter.end()

        if len(self.sigma['B']['th_ph']) > 0:
            sbz = np.mean(self.sigma['B']['z'])
            sbx = np.mean(self.sigma['B']['x'])
            sby = np.mean(self.sigma['B']['y'])
            sbi = sbz**2 + sbx**2 + sby**2
            painter = QPainter(self)
            painter.setFont(QFont('Arial', 14))
            painter.setPen(QColor(255, 255, 255))
            tq_width = int(self.width() * 3 / 2)
            y = int(0.25 * self.height() + 55)
            rect = QRect(0, 0, tq_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Σ< σ_Bi > = {sbi:.2f}")
            y = int(0.25 * self.height() + 90)
            rect = QRect(0, 0, tq_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< σ^By > = {sby:.2f}")
            y = int(0.25 * self.height() + 125)
            rect = QRect(0, 0, tq_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< σ^Bx > = {sbx:.2f}")
            y = int(0.25 * self.height() + 160)
            rect = QRect(0, 0, tq_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< σ^Bz > = {sbz:.2f}")
            y = int(0.25 * self.height() + 55)
            rect = QRect(0, y, tq_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Measurement: {self.measurementB}")
            y = int(0.25 * self.height() + 90)
            rect = QRect(0, y, tq_width, self.height() - y)
            painter.drawText(
                rect, Qt.AlignmentFlag.AlignCenter,
                f"Total Measurements: {len(self.sigma['B']['th_ph'])}")
            y = int(0.25 * self.height() + 125)
            rect = QRect(0, y, tq_width, self.height() - y)
            prob_p2 = self.count_p1B / len(self.sigma['B']['th_ph']) * 100
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< +1 > = {prob_p2:.1f}%")
            y = int(0.25 * self.height() + 160)
            rect = QRect(0, y, tq_width, self.height() - y)
            prob_m2 = self.count_m1B / len(self.sigma['B']['th_ph']) * 100
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< -1 > = {prob_m2:.1f}%")
            painter.end()

        if (len(self.sigma['A']['th_ph']) > 0) and \
                (len(self.sigma['B']['th_ph']) > 0) and \
                (len(self.sigma['A']['z']) > 3):
            # with limited number of measurements, the following
            # warning might appears.
            # RuntimeWarning: invalid value encountered in divide
            # c /= stddev[None, :]
            corrz = np.corrcoef(self.sigma['A']['z'],
                                self.sigma['B']['z'])[0, 1]
            corrx = np.corrcoef(self.sigma['A']['x'],
                                self.sigma['B']['x'])[0, 1]
            corry = np.corrcoef(self.sigma['A']['y'],
                                self.sigma['B']['y'])[0, 1]
            painter = QPainter(self)
            painter.setFont(QFont('Arial', 14))
            painter.setPen(QColor(255, 255, 255))
            y = int(0.25 * self.height() + 90)
            rect = QRect(0, 0, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Correlation <σ^Ay> <σ^By> = {corry:.2f}")
            y = int(0.25 * self.height() + 125)
            rect = QRect(0, 0, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Correlation <σ^Ax> <σ^Bx> = {corrx:.2f}")
            y = int(0.25 * self.height() + 160)
            rect = QRect(0, 0, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Correlation <σ^Az> <σ^Bz> = {corrz:.2f}")
            if (cfg.m):
                y = int(0.25 * self.height() + 55)
                corrm = np.corrcoef(self.sigma['A']['th_ph'],
                                    self.sigma['B']['th_ph'])[0, 1]
                rect = QRect(0, y, self.width(), self.height() - y)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                                 f"Correlation <σ^Am> <σ^Bm> = {corrm:.2f}")
            painter.end()
        glFlush()

    def drawRectangleAndArrow(self):
        glBegin(GL_QUADS)
        glVertex3f(-0.5, 0, -0.5)
        glVertex3f(0.5, 0, -0.5)
        glVertex3f(0.5, 0, 0.5)
        glVertex3f(-0.5, 0, 0.5)
        glEnd()

        # Draw an arrow
        glBegin(GL_LINES)
        glVertex3f(0., 0, -1)
        glVertex3f(0., 0, 1)
        glVertex3f(0., 0, 1)
        glVertex3f(-.2, 0, 0.75)
        glVertex3f(0., 0, 1)
        glVertex3f(.2, 0, 0.75)
        glVertex3f(-1, 0, 0)
        glVertex3f(1, 0, 0)
        glEnd()

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def update_rotation_thetaA(self, value: float):
        self.a_thetaA = np.deg2rad(value)
        self.update()

    def update_rotation_phiA(self, value: float):
        self.a_phiA = np.deg2rad(value)
        self.update()

    def measureA(self, current_state: np.ndarray):
        self.measureAB(current_state, True)

    def updateCountA(self, value: int):
        if value == 1:
            self.count_p1A += 1
            self.measurementA = 1
        else:
            self.count_m1A += 1
            self.measurementA = -1

    def measureAB(self, current_state: np.ndarray, measureA: bool):
        self.current_state = current_state
        for axis in ['z', 'x', 'y']:
            sp = self.measureSpin(self.directions[axis],
                                  self.directions[axis], True)
            self.sigma['A'][axis].append(sp[0])
            self.sigma['B'][axis].append(sp[1])

        # get the measurement direction for A
        directionAp = np.array([
            np.cos(self.a_thetaA * cfg.bloch_t / 2),
            np.exp(1j * self.a_phiA * cfg.bloch_p) * np.sin(
                self.a_thetaA * cfg.bloch_t / 2)])
        # get the opposite measurement direction
        directionAm = np.array([
            np.cos(np.pi / 2 + self.a_thetaA * cfg.bloch_t / 2),
            np.exp(1j * self.a_phiA * cfg.bloch_p) * np.sin(
                np.pi / 2 + self.a_thetaA * cfg.bloch_t / 2)])

        # get the measurement direction for B
        directionBp = np.array([
            np.cos(self.a_thetaB * cfg.bloch_t / 2),
            np.exp(1j * self.a_phiB * cfg.bloch_p) * np.sin(
                self.a_thetaB * cfg.bloch_t / 2)])
        # get the opposite measurement direction
        directionBm = np.array([
            np.cos(np.pi / 2 + self.a_thetaB * cfg.bloch_t / 2),
            np.exp(1j * self.a_phiB * cfg.bloch_p) * np.sin(
                np.pi / 2 + self.a_thetaB * cfg.bloch_t / 2)])

        if measureA:
            sp = self.measureSpin(np.array([directionAp, directionAm]),
                                  np.array([directionBp, directionBm]), True)
            self.sigma['A']['th_ph'].append(sp[0])
            self.updateCountA(sp[0])
            if cfg.m:
                self.sigma['B']['th_ph'].append(sp[1])
                self.updateCountB(sp[1])
        else:
            sp = self.measureSpin(np.array([directionAp, directionAm]),
                                  np.array([directionBp, directionBm]), False)
            self.sigma['B']['th_ph'].append(sp[1])
            self.updateCountB(sp[1])
            if cfg.m:
                self.sigma['A']['th_ph'].append(sp[0])
                self.updateCountA(sp[0])

        self.update()

    def update_rotation_thetaB(self, value: float):
        self.a_thetaB = np.deg2rad(value)
        self.update()

    def update_rotation_phiB(self, value: float):
        self.a_phiB = np.deg2rad(value)
        self.update()

    def measureB(self, current_state: np.ndarray):
        self.measureAB(current_state, False)

    def updateCountB(self, value: int):
        if value == 1:
            self.count_p1B += 1
            self.measurementB = 1
        else:
            self.count_m1B += 1
            self.measurementB = -1

    def measureSpin(self, directionsA: np.ndarray,
                    directionsB: np.ndarray, simulate_A: bool):
        '''
        Perform the measurement of the spin of two systems A and B,
        which one to simulate is decided by the caller.
        '''
        def RhoA(psi: np.ndarray):
            return np.outer(psi, psi.conj()).reshape(
                (2, 2, 2, 2)).trace(axis1=1, axis2=3)

        def RhoB(psi: np.ndarray):
            return np.outer(psi, psi.conj()).reshape((
                2, 2, 2, 2)).trace(axis1=0, axis2=2)

        psi = self.current_state
        # Define the projector operator for the "+1" state
        directionA_p1 = np.array(directionsA[0])
        directionA_m1 = np.array(directionsA[1])
        directionB_p1 = np.array(directionsB[0])
        directionB_m1 = np.array(directionsB[1])

        projectorA_p1 = np.outer(directionA_p1, directionA_p1.conj())
        projectorA_m1 = np.outer(directionA_m1, directionA_m1.conj())
        projectorB_p1 = np.outer(directionB_p1, directionB_p1.conj())
        projectorB_m1 = np.outer(directionB_m1, directionB_m1.conj())

        if simulate_A:
            # Create 4x4 projectors for the two-spin system
            projector_p1_s = np.kron(projectorA_p1, np.eye(2))
            projector_m1_s = np.kron(projectorA_m1, np.eye(2))
            projector_p11 = projectorA_p1
            projector_p21 = projectorB_p1

            # Calculate the reduced density matrix for the first spin
            # which is measured
            rho1 = RhoA(psi)
        else:
            projector_p1_s = np.kron(np.eye(2), projectorB_p1)
            projector_m1_s = np.kron(np.eye(2), projectorB_m1)
            projector_p11 = projectorB_p1
            projector_p21 = projectorA_p1
            rho1 = RhoB(psi)

        # Calculate the probability of this first spin being "+1"
        prob_p11 = np.linalg.norm(np.trace(np.dot(projector_p11, rho1)))
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
        rho2 = RhoB(psi_r) if simulate_A else RhoA(psi_r)

        # Calculate the probability of "system 2" being "+1"
        prob_p12 = np.linalg.norm(np.trace(np.dot(projector_p21, rho2)))

        # Generate a random number between 0 and 1
        random_number2 = random.uniform(0, 1)

        sp2 = 1 if random_number2 < prob_p12 else -1
        return (sp1, sp2) if simulate_A else (sp2, sp1)


class MainWindow(QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

        self.current_state = None
        self.simulation_thread = SimulationThread()
        self.simulation_thread.result.connect(self.store_simul_spin)
        self.simulation_thread.start()

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        match cfg.stype:
            case 1:
                desc = 'Product state: A = | u > B = | d >'
            case 2:
                desc = 'Product state: A = 1 / sqrt(2) * (| u > + | d >) ' \
                    'B = | u > / 2 + sqrt(3) / 2 * | d >'
            case 3:
                desc = 'Singlet state: '\
                    '| Psi > = 1 / sqrt(2) ( | ud > - | du > )'
            case 4:
                desc = 'Triplet state I: '\
                    '| Psi > = 1 / sqrt(2) ( | ud > + | du > )'
            case 5:
                desc = 'Triplet state II: '\
                    '| Psi > = 1 / sqrt(2) ( | uu > + | dd > )'
            case 6:
                desc = 'Triplet state III: '\
                    '| Psi > = 1 / sqrt(2) ( | uu > - | dd > )'
            case 7:
                desc = 'Partially entangled state: '\
                    '| Psi > = sqrt(0.6) | ud > - sqrt(0.4) | du >'

        self.setWindowTitle(f"Two quantum spin simulation - {desc}")

        self.opengl_widget = OpenGLWidget(self)

        self.slider1A = QSlider(Qt.Orientation.Horizontal, self)
        self.slider1A.setRange(0, 360)
        self.slider1A.valueChanged.connect(self.update_rotation_thetaA)
        self.label1A = QLabel("Apparatus A Rotation θ: 0", self)

        self.slider2A = QSlider(Qt.Orientation.Horizontal, self)
        self.slider2A.setRange(0, 360)
        self.slider2A.valueChanged.connect(self.update_rotation_phiA)
        self.label2A = QLabel("Apparatus A Rotation 𝜙: 0", self)

        self.buttonA = QPushButton('Make measurement system A', self)
        self.buttonA.clicked.connect(self.on_buttonA_clicked)

        self.slider1B = QSlider(Qt.Orientation.Horizontal, self)
        self.slider1B.setRange(0, 360)
        self.slider1B.valueChanged.connect(self.update_rotation_thetaB)
        self.label1B = QLabel("Apparatus B Rotation θ: 0", self)

        self.slider2B = QSlider(Qt.Orientation.Horizontal, self)
        self.slider2B.setRange(0, 360)
        self.slider2B.valueChanged.connect(self.update_rotation_phiB)
        self.label2B = QLabel("Apparatus B Rotation 𝜙: 0", self)

        self.buttonB = QPushButton('Make measurement system B', self)
        self.buttonB.clicked.connect(self.on_buttonB_clicked)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.opengl_widget)
        self.gridlayout = QGridLayout()
        self.gridlayout.addWidget(self.slider1A, 0, 0)
        self.gridlayout.addWidget(self.label1A, 0, 1)
        self.gridlayout.addWidget(self.slider2A, 1, 0)
        self.gridlayout.addWidget(self.label2A, 1, 1)
        self.gridlayout.addWidget(self.buttonA, 2, 0, 1, 2)
        self.gridlayout.addWidget(self.slider1B, 3, 0)
        self.gridlayout.addWidget(self.label1B, 3, 1)
        self.gridlayout.addWidget(self.slider2B, 4, 0)
        self.gridlayout.addWidget(self.label2B, 4, 1)
        self.gridlayout.addWidget(self.buttonB, 5, 0, 1, 2)
        self.layout.addLayout(self.gridlayout)

    def store_simul_spin(self, current_simul_state: np.ndarray):
        self.current_state = current_simul_state

    def update_rotation_thetaA(self, value: float):
        self.opengl_widget.update_rotation_thetaA(value)
        self.label1A.setText(f"Apparatus A Rotation θ: {value}")

    def update_rotation_phiA(self, value: float):
        self.opengl_widget.update_rotation_phiA(value)
        self.label2A.setText(f"Apparatus A Rotation 𝜙: {value}")

    def on_buttonA_clicked(self):
        self.opengl_widget.measureA(self.current_state)

    def update_rotation_thetaB(self, value: float):
        self.opengl_widget.update_rotation_thetaB(value)
        self.label1B.setText(f"Apparatus B Rotation θ: {value}")

    def update_rotation_phiB(self, value: float):
        self.opengl_widget.update_rotation_phiB(value)
        self.label2B.setText(f"Apparatus B Rotation 𝜙: {value}")

    def on_buttonB_clicked(self):
        self.opengl_widget.measureB(self.current_state)


class CustomHelpFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        # Preserve line breaks by not wrapping text
        return "\n".join([indent + line for line in text.splitlines()])


def parse_color(color_string):
    """Parse a comma-separated RGB string and normalize it
    to a tuple of floats."""
    rgb = tuple(int(x) for x in color_string.split(','))
    return tuple(c / 255.0 for c in rgb)


def main():
    # Set a fixed seed value
    seed_value = 5948
    random.seed(seed_value)
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=CustomHelpFormatter)
    parser.add_argument('-t', '--simul_type', help='simulation type',
                        required=False)
    parser.add_argument('-m', '--measure_both', action='store_true',
                        help='Measure both systems', required=False)
    parser.add_argument('-l', '--color_left', type=parse_color,
                        help='Set the left apparatus color as comma-separated'
                        ' RGB values (0-255). Example: -l 0,255,0 -'
                        ' Default: green')
    parser.add_argument('-r', '--color_right', type=parse_color,
                        help='Set the right apparatus color as comma-separated'
                        ' RGB values (0-255). Example: -r 255,0,0 -'
                        ' Default: red')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose output', required=False)
    parser.add_argument('-b', '--bloch-theta', type=float,
                        help='coefficient theta between real '
                        'and Hilbert world')
    parser.add_argument('-c', '--bloch-phi', type=float,
                        help='coefficient phi between real '
                        'and Hilbert world')
    args = parser.parse_args()
    if (args.simul_type):
        cfg.stype = int(args.simul_type)
    if (args.measure_both):
        cfg.m = True
    if (args.color_left):
        cfg.color_left = args.color_left
    if (args.color_right):
        cfg.color_right = args.color_right
    if (args.verbose):
        cfg.verbose = True
    if (args.bloch_theta):
        cfg.bloch_t = args.bloch_theta
    if (args.bloch_phi):
        cfg.bloch_p = args.bloch_phi
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    main()
