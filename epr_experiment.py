#!/usr/bin/env python3
'''
/************************/
/*  epr_experiment.py   */
/*    Version 1.0       */
/*      2024/05/12      */
/************************/
'''
import argparse
import math
from mod_spin_operators import TwoSpin
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QPushButton, QLabel
from PyQt6.QtWidgets import QButtonGroup, QRadioButton
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QFont, QColor
from PyQt6.QtCore import QRect
from PyQt6.QtCore import Qt
from OpenGL.GL import (
    glClear, glClearColor, glEnable, glPushMatrix, glPopMatrix, glRotatef,
    glTranslatef, glBegin, glEnd, glVertex3f, glViewport, glMatrixMode,
    glLoadIdentity, glColor3f, glLineWidth, glHint, glScalef)
from OpenGL.GLU import gluPerspective, gluLookAt
from OpenGL.GLUT import glutStrokeCharacter, GLUT_STROKE_ROMAN
from OpenGL.GL import (
    GL_DEPTH_TEST, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_QUADS, GL_LINES, glFlush, GL_PROJECTION, GL_MODELVIEW,
    GL_TRIANGLE_FAN, GL_LINE_SMOOTH, GL_LINE_SMOOTH_HINT,
    GL_NICEST, GL_BLEND)
import random
import sys
from types import SimpleNamespace

cfg = SimpleNamespace(
    stype=1, n=100, color_up=[0, 1, 0], color_down=[1, 0, 0],
    invert=True, theta1=0, phi1=0, theta2=0, phi2=0)

description = (
    'This script simulates two entangled spin following '
    'quantum mechanics principles.\n\n'
    'It can be used to simulate the violation of Bell\'s theorem'
    'and therefore the impossibility of the presence of hidden variables.\n\n'
    'Simulation types available (-t SIMUL_TYPE, --simul_type SIMUL_TYPE):\n'
    '1 - Singlet state\n'
    '    | Psi > = 1 / sqrt(2) * (| ud > - | du >) [DEFAULT]\n'
    '2 - Triplet state I\n'
    '    | Psi > = 1 / sqrt(2) * (| ud > + | du >)\n'
    '3 - Triplet state II\n'
    '    | Psi > = 1 / sqrt(2) * (| uu > + | dd >)\n'
    '4 - Triplet state III\n'
    '    | Psi > = 1 / sqrt(2) * (| uu > - | dd >)\n'
    'Both apparatus measure at the same time.\n'
    'There is a button which allow a random selection of the direction, so '
    'that statistically they will measure the same direction ⅓ of the times.\n'
    ' There is a button to perform \'n\' measurements, with the number that '
    'can be set with the command line option '
    '"-m, --measurement_number" (default = 100).\n\n'
    'It is possible to set the color for the spin up (| +1 >) result '
    'with the command line option "-u, --color_up" (default = green) '
    'and for the spin down (| -1 >) with '
    '"-d, --color_down (default = red).\n\n'
    'By default the results are inverted, so, in the case of singlet, if '
    'the apparatus 1 measure | +1 >, the apparatus 2 will also agree 100% '
    'of the time if oriented in the same direction, otherwise will be a '
    '0% agreement. It is set in this way for onvenience of analyizing '
    'the results and can be overwritten with the command line option '
    '-n --no-invert.\n\n'
    'The orientation of the apparatus can be set with theta1, theta2, '
    'phi1 and phi2 in degrees (default set to 0).\n\n'
    'The equivalent result (100% agreement if in the same direction '
    'and 25% otherwise with the following configurations:\n'
    '1 - invert = True - theta2 = 0° (both apparatus same direction).\n'
    '2 - invert = False - theta2 = 180° (second apparatus upside down).\n'
)


class OpenGLWidget(QOpenGLWidget):

    def __init__(self, parent):
        super(OpenGLWidget, self).__init__(parent)
        self.button1 = None
        self.button1fix = None
        self.button2 = None
        self.button2fix = None
        self.isFixed = None
        self.measurement1 = None
        self.measurement2 = None
        self.measurements1 = np.array([], dtype=int)
        self.measurements2 = np.array([], dtype=int)
        self.switches1 = np.array([], dtype=int)
        self.switches2 = np.array([], dtype=int)

        # Set the Spin Type
        spin = TwoSpin()
        match cfg.stype:
            case 1:
                spin.Singlet()
            case 2:
                spin.Triplet(1)
            case 3:
                spin.Triplet(2)
            case 4:
                spin.Triplet(3)
            case _:
                raise ValueError(
                    f"Incorrect simulation type {self.simul_type}")
        self.current_state = spin.psi

        # Initialize the directions which have a 120° rotation between
        # each other
        # First apparatus
        theta_i = (cfg.theta1 + 240) * np.pi / 180
        theta_j = (cfg.theta1) * np.pi / 180
        theta_k = (cfg.theta1 + 120) * np.pi / 180
        phi_i = cfg.phi1 * np.pi / 180
        self.direction1p = np.array([
            [
                np.cos(theta_i / 2),
                np.exp(1j * phi_i) * np.sin(theta_i / 2)
            ],
            [
                np.cos(theta_j / 2),
                np.exp(1j * phi_i) * np.sin(theta_j / 2)
            ],
            [
                np.cos(theta_k / 2),
                np.exp(1j * phi_i) * np.sin(theta_k / 2)
            ]
        ])
        theta_i = (cfg.theta1 + 240) * np.pi / 180 + np.pi
        theta_j = (cfg.theta1) * np.pi / 180 + np.pi
        theta_k = (cfg.theta1 + 120) * np.pi / 180 + np.pi
        phi_i = cfg.phi1 * np.pi / 180
        self.direction1m = np.array([
            [
                np.cos(theta_i / 2),
                np.exp(1j * phi_i) * np.sin(theta_i / 2)
            ],
            [
                np.cos(theta_j / 2),
                np.exp(1j * phi_i) * np.sin(theta_j / 2)
            ],
            [
                np.cos(theta_k / 2),
                np.exp(1j * phi_i) * np.sin(theta_k / 2)
            ]
        ])
        # Second apparatus
        theta_i = (cfg.theta2 + 240) * np.pi / 180
        theta_j = (cfg.theta2) * np.pi / 180
        theta_k = (cfg.theta2 + 120) * np.pi / 180
        phi_i = cfg.phi2 * np.pi / 180
        self.direction2p = np.array([
            [
                np.cos(theta_i / 2),
                np.exp(1j * phi_i) * np.sin(theta_i / 2)
            ],
            [
                np.cos(theta_j / 2),
                np.exp(1j * phi_i) * np.sin(theta_j / 2)
            ],
            [
                np.cos(theta_k / 2),
                np.exp(1j * phi_i) * np.sin(theta_k / 2)
            ]
        ])

        theta_i = (cfg.theta2 + 240) * np.pi / 180 + np.pi
        theta_j = (cfg.theta2) * np.pi / 180 + np.pi
        theta_k = (cfg.theta2 + 120) * np.pi / 180 + np.pi
        phi_i = cfg.phi2 * np.pi / 180
        self.direction2m = np.array([
            [
                np.cos(theta_i / 2),
                np.exp(1j * phi_i) * np.sin(theta_i / 2)
            ],
            [
                np.cos(theta_j / 2),
                np.exp(1j * phi_i) * np.sin(theta_j / 2)
            ],
            [
                np.cos(theta_k / 2),
                np.exp(1j * phi_i) * np.sin(theta_k / 2)
            ]
        ])

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_BLEND)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # Adjust the camera view
        gluLookAt(0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glTranslatef(-2.5, 0.0, 0.0)
        self.drawApparatus(True)
        glTranslatef(5.0, 0.0, 0.0)
        self.drawApparatus(False)
        self.drawText()
        glFlush()

    def drawApparatus(self, app1: bool):
        glMatrixMode(GL_MODELVIEW)
        self.drawApparatusB()
        self.drawLetters()
        self.drawResults(app1)
        button = self.button1 if app1 else self.button2
        if button is not None:
            self.drawButtonSelected(button)
        self.drawApparatusW()
        self.drawArrows(button, app1)

    def drawApparatusB(self):
        glPushMatrix()
        glColor3f(0.18, 0.18, 0.18)
        glLineWidth(2.5)
        glBegin(GL_LINES)
        glVertex3f(-1.5, 0, -0.5)
        glVertex3f(1.5, 0, -0.5)
        glVertex3f(-1.5, 0, 0.5)
        glVertex3f(-0.1, 0, 0.5)
        glVertex3f(1.5, 0, 0.5)
        glVertex3f(0.1, 0, 0.5)
        glVertex3f(0.1, 0, 0.5)
        glVertex3f(0.1, 0, 0.6)
        glVertex3f(-0.1, 0, 0.5)
        glVertex3f(-0.1, 0, 0.6)
        glVertex3f(-1.5, 0, -0.5)
        glVertex3f(-1.5, 0, 0.5)
        glVertex3f(1.5, 0, -0.5)
        glVertex3f(1.5, 0, 0.5)
        glVertex3f(0.5, 0, -0.5)
        glVertex3f(0.5, 0, 0.5)
        glVertex3f(-0.5, 0, -0.5)
        glVertex3f(-0.5, 0, 0.5)
        glEnd()
        glPopMatrix()

    def drawLetters(self):
        glLineWidth(5)
        glColor3f(0.18, 0.18, 0.18)
        glPushMatrix()
        glTranslatef(-1.05, 0.0, -0.1)
        glRotatef(90, 1.0, 0.0, 0.0)
        glScalef(0.0025, 0.0025, 0.0025)
        glutStrokeCharacter(GLUT_STROKE_ROMAN, ord("L"))
        glPopMatrix()
        glPushMatrix()
        glTranslatef(-0.05, 0., -0.1)
        glRotatef(90, 1.0, 0.0, 0.0)
        glScalef(0.0025, 0.0025, 0.0025)
        glutStrokeCharacter(GLUT_STROKE_ROMAN, ord("C"))
        glPopMatrix()
        glPushMatrix()
        glTranslatef(0.9, 0., -0.1)
        glRotatef(90, 1.0, 0.0, 0.0)
        glScalef(0.0025, 0.0025, 0.0025)
        glutStrokeCharacter(GLUT_STROKE_ROMAN, ord("R"))
        glPopMatrix()

    def drawResults(self, app1: bool):
        glPushMatrix()
        measurement = self.measurement1 if app1 else self.measurement2
        if measurement is not None:
            if measurement == 1:
                glColor3f(*cfg.color_up)
            else:
                glColor3f(*cfg.color_down)
        else:
            glColor3f(0.38, 0.38, 0.38)
        center_x, center_z = 0, 1.1
        radius = 0.4
        sides = 100
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(center_x, 0, center_z)
        for i in range(sides + 1):
            angle = 2 * math.pi * i / sides
            x = center_x + radius * math.cos(angle)
            z = center_z + radius * math.sin(angle)
            glVertex3f(x, 0, z)
        glEnd()
        glPopMatrix()

    def drawButtonSelected(self, n: int):
        glPushMatrix()
        glColor3f(1, 0.6, 0.2)
        glBegin(GL_QUADS)
        glVertex3f(-1.45 + 1 * n, 0, -0.45)
        glVertex3f(-0.55 + 1 * n, 0, -0.45)
        glVertex3f(-0.55 + 1 * n, 0, 0.45)
        glVertex3f(-1.45 + 1 * n, 0, 0.45)
        glEnd()
        glPopMatrix()

    def drawApparatusW(self):
        glPushMatrix()
        glColor3f(0.97, 0.97, 0.97)
        glBegin(GL_QUADS)
        glVertex3f(-1.5, 0, -0.5)
        glVertex3f(1.5, 0, -0.5)
        glVertex3f(1.5, 0, 0.5)
        glVertex3f(-1.5, 0, 0.5)
        glEnd()
        glBegin(GL_QUADS)
        glVertex3f(-0.1, 0, 0.5)
        glVertex3f(0.1, 0, 0.5)
        glVertex3f(0.1, 0, 1.2)
        glVertex3f(-0.1, 0, 1.2)
        glEnd()
        center_x, center_z = 0, 1.1
        radius = 0.5
        sides = 100
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(center_x, 0, center_z)
        for i in range(sides + 1):
            angle = 2 * math.pi * i / sides
            x = center_x + radius * math.cos(angle)
            z = center_z + radius * math.sin(angle)
            glVertex3f(x, 0, z)
        glEnd()
        glPopMatrix()

    def drawArrows(self, n, app1: bool):
        theta = cfg.theta1 if app1 else cfg.theta2
        phi = cfg.phi1 if app1 else cfg.phi2
        glPushMatrix()
        if n == 1:
            glColor3f(1, 0.6, 0.2)
        else:
            glColor3f(0.97, 0.97, 0.97)
        glRotatef(-phi, 0.0, 0.0, 1.0)
        glRotatef(theta, 0, 1, 0)
        glBegin(GL_LINES)
        glVertex3f(0., 0, 0)
        glVertex3f(0., 0, 2.2)
        glVertex3f(0., 0, 2.19)
        glVertex3f(-0.2, 0, 1.9)
        glVertex3f(0., 0, 2.19)
        glVertex3f(0.2, 0, 1.9)
        glEnd()
        glPopMatrix()
        glPushMatrix()
        if n == 0:
            glColor3f(1, 0.6, 0.2)
        else:
            glColor3f(0.97, 0.97, 0.97)
        glRotatef(-phi, 0.0, 0.0, 1.0)
        glRotatef(240 + theta, 0, 1, 0)
        glBegin(GL_LINES)
        glVertex3f(0., 0, 0)
        glVertex3f(0., 0, 2.2)
        glVertex3f(0., 0, 2.19)
        glVertex3f(-0.2, 0, 1.9)
        glVertex3f(0., 0, 2.19)
        glVertex3f(0.2, 0, 1.9)
        glEnd()
        glPopMatrix()
        glPushMatrix()
        if n == 2:
            glColor3f(1, 0.6, 0.2)
        else:
            glColor3f(0.97, 0.97, 0.97)
        glRotatef(-phi, 0.0, 0.0, 1.0)
        glRotatef(120 + theta, 0, 1, 0)
        glBegin(GL_LINES)
        glVertex3f(0., 0, 0)
        glVertex3f(0., 0, 2.2)
        glVertex3f(0., 0, 2.19)
        glVertex3f(-0.2, 0, 1.9)
        glVertex3f(0., 0, 2.19)
        glVertex3f(0.2, 0, 1.9)
        glEnd()
        glPopMatrix()

    def drawText(self):
        painter = QPainter(self)
        painter.setFont(QFont('Arial', 14))
        step = 35
        base1 = 125
        base2 = 115
        painter.setPen(QColor(255, 153, 51))
        y = int(0.25 * self.height() + base1 + 5 * step)
        rect = QRect(0, 0, 150, self.height() - y)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Apparatus 1")
        y = int(0.25 * self.height() + base1 + 4 * step)
        rect = QRect(0, 0, 150, self.height() - y)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                         f"Rotation θ: {cfg.theta1:.1f}°")
        y = int(0.25 * self.height() + base1 + 3 * step)
        rect = QRect(0, 0, 150, self.height() - y)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                         f"Rotation 𝜙: {cfg.phi1:.1f}°")
        y = int(0.25 * self.height() + base1 + 5 * step)
        rect = QRect(self.width() - 150, 0, 150, self.height() - y)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Apparatus 2")
        y = int(0.25 * self.height() + base1 + 4 * step)
        rect = QRect(self.width() - 150, 0, 150, self.height() - y)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                         f"Rotation θ: {cfg.theta2:.1f}°")
        y = int(0.25 * self.height() + base1 + 3 * step)
        rect = QRect(self.width() - 150, 0, 150, self.height() - y)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                         f"Rotation 𝜙: {cfg.phi2:.1f}°")
        if self.measurement1:
            painter.setPen(QColor(255, 255, 255))
            half_width = int(self.width() / 2)
            tq_width = int(self.width() * 3 / 2)

            measurements_nb = len(self.measurements1)
            y = int(0.25 * self.height() + base1 + 0 * step)
            rect = QRect(0, y, half_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Measurement: {self.measurement1}")
            y = int(0.25 * self.height() + base1 + 1 * step)
            rect = QRect(0, y, half_width, self.height() - y)
            prob_p1 = np.count_nonzero(
                self.measurements1 == 1) / measurements_nb
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< color 1 > = {prob_p1*100:.1f}%")
            y = int(0.25 * self.height() + base1 + 2 * step)
            rect = QRect(0, y, half_width, self.height() - y)
            prob_m1 = np.count_nonzero(
                self.measurements1 == -1) / measurements_nb
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< color 2 > = {prob_m1*100:.1f}%")

            y = int(0.25 * self.height() + base1 + 0 * step)
            rect = QRect(0, y, tq_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Measurement: {self.measurement2}")
            y = int(0.25 * self.height() + base1 + 1 * step)
            rect = QRect(0, y, tq_width, self.height() - y)
            prob_p2 = np.count_nonzero(
                self.measurements2 == 1) / measurements_nb
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< color 1 > = {prob_p2*100:.1f}%")
            y = int(0.25 * self.height() + base1 + 2 * step)
            rect = QRect(0, y, tq_width, self.height() - y)
            prob_m2 = np.count_nonzero(
                self.measurements2 == -1) / measurements_nb
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< color 2 > = {prob_m2*100:.1f}%")

            same_mask = self.switches1 == self.switches2
            diff_mask = self.switches1 != self.switches2
            num_same = np.sum(same_mask)
            num_diff = np.sum(diff_mask)
            # Count occurrences where measurements have
            # the same value for same_mask
            if num_same > 0:
                equal_same_mask = np.sum(
                    self.measurements1[same_mask] == self.measurements2[
                        same_mask])

            # Count occurrences where measurements have
            # the same value for diff_mask
            if num_diff > 0:
                equal_diff_mask = np.sum(
                    self.measurements1[diff_mask] == self.measurements2[
                        diff_mask])

            y = int(0.25 * self.height() + base1 + 5 * step)
            rect = QRect(0, 0, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "Same Switch")
            y = int(0.25 * self.height() + base1 + 4 * step)
            rect = QRect(0, 0, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "Percentage = "
                             f"{num_same/measurements_nb*100:.1f}%")
            if num_same > 0:
                y = int(0.25 * self.height() + base1 + 3 * step)
                rect = QRect(0, 0, self.width(), self.height() - y)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                                 "% same results = "
                                 f"{equal_same_mask/num_same*100:.1f}%")
            y = int(0.25 * self.height() + base2 + 2 * step)
            rect = QRect(0, 0, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "Different Switch")
            y = int(0.25 * self.height() + base2 + 1 * step)
            rect = QRect(0, 0, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "Percentage = "
                             f"{num_diff/measurements_nb*100:.1f}%")
            if num_diff > 0:
                y = int(0.25 * self.height() + base2 + 0 * step)
                rect = QRect(0, 0, self.width(), self.height() - y)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                                 "% same results = "
                                 f"{equal_diff_mask/num_diff*100:.1f}%")
            y = int(0.25 * self.height() + base1 + 0 * step)
            rect = QRect(0, y, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "Total Measurements")
            y = int(0.25 * self.height() + base1 + 1 * step)
            rect = QRect(0, y, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"{measurements_nb}")

        painter.end()
        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def measure(self, n):
        for i in range(n):
            # Randomize the button is selected
            if not self.isFixed:
                self.button1 = random.randint(0, 2)
                self.button2 = random.randint(0, 2)

            self.switches1 = np.append(self.switches1, self.button1)
            self.switches2 = np.append(self.switches2, self.button2)
            # it is possible to measure always the first apparatus first
            # but to give more variability, which one to measure is
            # randomly chosen
            random_number = random.uniform(0, 1)
            simulate_1 = True if random_number < 0.5 else False
            self.measurement1, self.measurement2 = self.measureSpin(
                np.array([self.direction1p[self.button1],
                          self.direction1m[self.button1]]),
                np.array([self.direction2p[self.button2],
                          self.direction2m[self.button2]]),
                simulate_1)

            # Invert the results for apparatus 2 if in the config,
            # so, in the case of singlet, if the apparatus 1 measure +1,
            #  apparatus 2 will agree 100% of the time it is oriented
            # in the same direction.
            if cfg.invert:
                self.measurement2 *= -1
            self.measurements1 = np.append(
                self.measurements1, self.measurement1)
            self.measurements2 = np.append(
                self.measurements2, self.measurement2)

        # redraw
        self.update()

    def measureSpin(self, directions1: np.ndarray,
                    directions2: np.ndarray, simulate_1: bool):
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

        psi = self.current_state
        # Define the projector operator for the "+1" state
        direction1_p1 = np.array(directions1[0])
        direction1_m1 = np.array(directions1[1])
        direction2_p1 = np.array(directions2[0])
        direction2_m1 = np.array(directions2[1])
        projectorA_p1 = np.outer(direction1_p1, direction1_p1.conj())
        projectorA_m1 = np.outer(direction1_m1, direction1_m1.conj())
        projectorB_p1 = np.outer(direction2_p1, direction2_p1.conj())
        projectorB_m1 = np.outer(direction2_m1, direction2_m1.conj())

        if simulate_1:
            # Create 4x4 projectors for the two-spin system
            projector_p1_s = np.kron(projectorA_p1, np.eye(2))
            projector_m1_s = np.kron(projectorA_m1, np.eye(2))
            projector_p11 = projectorA_p1
            projector_p21 = projectorB_p1

            # Calculate the reduced density matrix for the first spin
            # which is measured
            rho_i = Rho1(psi)
        else:
            projector_p1_s = np.kron(np.eye(2), projectorB_p1)
            projector_m1_s = np.kron(np.eye(2), projectorB_m1)
            projector_p11 = projectorB_p1
            projector_p21 = projectorA_p1
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
        return (sp1, sp2)

    def update_button1(self, value: int):
        self.button1 = value
        self.button1fix = value
        self.update()

    def update_button2(self, value: int):
        self.button2 = value
        self.button2fix = value
        self.update()

    def update_button3(self, value: int):
        if value == 0:
            self.isFixed = True
            self.button1 = self.button1fix
            self.button2 = self.button2fix
        else:
            self.isFixed = False
        self.update()


class MainWindow(QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        match cfg.stype:
            case 1:
                desc = 'Singlet state: '\
                    '| Psi > = 1 / sqrt(2) ( | ud > - | du > )'
            case 2:
                desc = 'Triplet state I: '\
                    '| Psi > = 1 / sqrt(2) ( | ud > + | du > )'
            case 3:
                desc = 'Triplet state II: '\
                    '| Psi > = 1 / sqrt(2) ( | uu > + | dd > )'
            case 4:
                desc = 'Triplet state III: '\
                    '| Psi > = 1 / sqrt(2) ( | uu > - | dd > )'

        self.setWindowTitle(f"EPR Experiment - {desc}")

        self.opengl_widget = OpenGLWidget(self)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.opengl_widget)
        self.gridlayout = QGridLayout()
        self.radioButtonsLCR1 = QButtonGroup(self)
        self.radioButtonL1 = QRadioButton("L")
        self.radioButtonC1 = QRadioButton("C")
        self.radioButtonR1 = QRadioButton("R")
        self.radioButtonsLCR1.addButton(self.radioButtonL1, 0)
        self.radioButtonsLCR1.addButton(self.radioButtonC1, 1)
        self.radioButtonsLCR1.addButton(self.radioButtonR1, 2)
        self.radioButtonsLCR1.idToggled.connect(self.radio_button1_toggled)
        self.labelLeftApparatusSwitch1 = QLabel(
            "Apparatus 1 switch\nif \"Fix\" is selected")
        self.hboxLCR1 = QHBoxLayout()
        self.hboxLCR1.addWidget(self.radioButtonL1)
        self.hboxLCR1.addWidget(self.radioButtonC1)
        self.hboxLCR1.addWidget(self.radioButtonR1)
        self.containerLCR1 = QWidget()
        self.containerLCR1.setLayout(self.hboxLCR1)

        self.radioButtonsLCR2 = QButtonGroup(self)
        self.radioButtonL2 = QRadioButton("L")
        self.radioButtonC2 = QRadioButton("C")
        self.radioButtonR2 = QRadioButton("R")
        self.radioButtonsLCR2.addButton(self.radioButtonL2, 0)
        self.radioButtonsLCR2.addButton(self.radioButtonC2, 1)
        self.radioButtonsLCR2.addButton(self.radioButtonR2, 2)
        self.radioButtonsLCR2.idToggled.connect(self.radio_button2_toggled)
        self.labelLeftApparatusSwitch2 = QLabel(
            "Apparatus 2 switch\nif \"Fix\" is selected")
        self.hboxLCR2 = QHBoxLayout()
        self.hboxLCR2.addWidget(self.radioButtonL2)
        self.hboxLCR2.addWidget(self.radioButtonC2)
        self.hboxLCR2.addWidget(self.radioButtonR2)
        self.containerLCR2 = QWidget()
        self.containerLCR2.setLayout(self.hboxLCR2)

        self.radioButtonsFixRandom = QButtonGroup(self)
        self.radioButtonFix = QRadioButton("Fix")
        self.radioButtonRandom = QRadioButton("Random")
        self.radioButtonsFixRandom.addButton(self.radioButtonFix, 0)
        self.radioButtonsFixRandom.addButton(self.radioButtonRandom, 1)
        self.radioButtonsFixRandom.idToggled.connect(
            self.radio_button3_toggled)
        self.labelSwitchSetting = QLabel("Switch setting")
        self.hboxFixRandom = QHBoxLayout()
        self.hboxFixRandom.addWidget(self.radioButtonFix)
        self.hboxFixRandom.addWidget(self.radioButtonRandom)
        self.containerFixRandom = QWidget()
        self.containerFixRandom.setLayout(self.hboxFixRandom)

        self.button1 = QPushButton('Measure (single)', self)
        self.button1.clicked.connect(self.on_button1_clicked)
        self.button2 = QPushButton(
            f'Measure ({cfg.n} times)', self)
        self.button2.clicked.connect(self.on_button2_clicked)

        self.containerLCR1.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        self.gridlayout.addWidget(self.containerLCR1, 0, 1)
        self.gridlayout.addWidget(self.labelLeftApparatusSwitch1, 0, 0)
        self.gridlayout.setColumnStretch(0, 0)
        self.gridlayout.setRowStretch(0, 0)
        self.containerLCR2.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        self.gridlayout.addWidget(self.containerLCR2, 0, 2)
        self.gridlayout.addWidget(self.labelLeftApparatusSwitch2, 0, 3)

        self.containerFixRandom.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        self.gridlayout.addWidget(self.containerFixRandom, 1, 1)
        self.gridlayout.addWidget(self.labelSwitchSetting, 1, 0)
        self.gridlayout.addWidget(self.button1, 1, 2)
        self.gridlayout.addWidget(self.button2, 1, 3)
        self.gridlayout.setColumnStretch(1, 0)
        self.gridlayout.setRowStretch(1, 0)

        # Set Default
        self.radioButtonC1.setChecked(True)
        self.radioButtonC2.setChecked(True)
        self.radioButtonFix.setChecked(True)

        self.layout.addLayout(self.gridlayout)

    def radio_button1_toggled(self, id, checked):
        if checked:
            self.opengl_widget.update_button1(id)

    def radio_button2_toggled(self, id, checked):
        if checked:
            self.opengl_widget.update_button2(id)

    def radio_button3_toggled(self, id, checked):
        if checked:
            self.opengl_widget.update_button3(id)

    def on_button1_clicked(self):
        self.opengl_widget.measure(1)

    def on_button2_clicked(self):
        self.opengl_widget.measure(cfg.n)


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
    seed_value = 9285
    random.seed(seed_value)
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=CustomHelpFormatter)
    parser.add_argument('-t', '--simul_type', help='simulation type',
                        required=False)
    parser.add_argument('-m', '--measurement_number', type=int, default=100,
                        help='Number of simultaneous measurements - '
                        'Default: 100')
    parser.add_argument('-n', '--no-invert', action='store_true',
                        help='Do not invert the results for apparatus 2',
                        required=False)
    parser.add_argument('-u', '--color_up', type=parse_color,
                        help='Set the spin up (| +1 >) color as '
                        'comma-separated RGB values (0-255). '
                        'Example: -c 0,255,0 - Default: green')
    parser.add_argument('-d', '--color_down', type=parse_color,
                        help='Set the spin down (| -1 >) color as '
                        'comma-separated RGB values (0-255). '
                        'Example: -c 255,0,0 - Default: red')
    parser.add_argument('-r', '--theta1', type=float,
                        help='angle theta1 in degrees')
    parser.add_argument('-s', '--theta2', type=float,
                        help='angle theta2 in degrees')
    parser.add_argument('-p', '--phi1', type=float,
                        help='angle phi1 in degrees')
    parser.add_argument('-q', '--phi2', type=float,
                        help='angle phi2 in degrees')

    args = parser.parse_args()
    if (args.simul_type):
        cfg.stype = int(args.simul_type)
    if (args.measurement_number):
        cfg.n = int(args.measurement_number)
    if (args.no_invert):
        cfg.invert = False
    if (args.color_up):
        cfg.color_up = args.color_up
    if (args.color_down):
        cfg.color_down = args.color_down
    if (args.theta1):
        cfg.theta1 = args.theta1
    if (args.theta2):
        cfg.theta2 = args.theta2
    if (args.phi1):
        cfg.phi1 = args.phi1
    if (args.phi2):
        cfg.phi2 = args.phi2
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    main()
