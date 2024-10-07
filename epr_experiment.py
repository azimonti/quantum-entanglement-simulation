#!/usr/bin/env python3
'''
/************************/
/*  epr_experiment.py   */
/*    Version 1.1       */
/*      2024/05/28      */
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
    invert=True, theta1=0, phi1=0, theta2=0, phi2=0,
    # additional coefficients  to to convert the real-space angles
    # into the corresponding angles in the Hilbert space (Bloch sphere)
    bloch_t=1.0, bloch_p=1.0,
    appthetaL=240, appthetaC=0, appthetaR=120, experiment=-1,
    verbose=False)

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
    '2 - invert = False - theta2 = 180° (second apparatus upside down).\n\n'
    'For convenience, two set of experiments can be selected with the '
    'command line option "-e, --experiment", and the variables will be set '
    'automatically:\n'
    '1 - The detectors are three Stern-Gerlach magnets one oriented along '
    'the z axis and the other two in the zx plane with ±120° rotation.\n'
    '    The particles are two entangled electrons in the singlet state.\n'
    '2 - The apparatus is composed by two polarizers which send two photons '
    'to three photodectors, one oriented along the z axis and the other two '
    'in the zx plane with 22.5° and 45° rotation.\n'
    '    The particles are two entangled photons in the second triplet '
    ' state | Psi > = 1 / sqrt(2) ( | uu > + | dd > ).'
    'Selecting either of these experiments will ignore any physical variable '
    'set from the command line (e.g. theta1, theta2, ..).\n'
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
        self.spin = spin
        # Initialize the directions which have a defined rotation between
        # each other
        # First apparatus
        theta_i = (cfg.theta1 + cfg.appthetaL) * cfg.bloch_t * np.pi / 180
        theta_j = (cfg.theta1 + cfg.appthetaC) * cfg.bloch_t * np.pi / 180
        theta_k = (cfg.theta1 + cfg.appthetaR) * cfg.bloch_t * np.pi / 180
        phi_i = cfg.phi1 * cfg.bloch_p * np.pi / 180
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
        theta_i = (
            cfg.theta1 + cfg.appthetaL) * cfg.bloch_t * np.pi / 180 + np.pi
        theta_j = (
            cfg.theta1 + cfg.appthetaC) * cfg.bloch_t * np.pi / 180 + np.pi
        theta_k = (
            cfg.theta1 + cfg.appthetaR) * cfg.bloch_t * np.pi / 180 + np.pi
        phi_i = cfg.phi1 * cfg.bloch_p * np.pi / 180
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
        theta_i = (cfg.theta2 + cfg.appthetaL) * cfg.bloch_t * np.pi / 180
        theta_j = (cfg.theta2 + cfg.appthetaC) * cfg.bloch_t * np.pi / 180
        theta_k = (cfg.theta2 + cfg.appthetaR) * cfg.bloch_t * np.pi / 180
        phi_i = cfg.phi2 * cfg.bloch_p * np.pi / 180
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
        theta_i = (
            cfg.theta2 + cfg.appthetaL) * cfg.bloch_t * np.pi / 180 + np.pi
        theta_j = (
            cfg.theta2 + cfg.appthetaC) * cfg.bloch_t * np.pi / 180 + np.pi
        theta_k = (
            cfg.theta2 + cfg.appthetaR) * cfg.bloch_t * np.pi / 180 + np.pi
        phi_i = cfg.phi2 * cfg.bloch_p * np.pi / 180
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
        if n == 0:
            glColor3f(1, 0.6, 0.2)
        else:
            glColor3f(0.97, 0.97, 0.97)
        glRotatef(-phi, 0.0, 0.0, 1.0)
        glRotatef(cfg.appthetaL + theta, 0, 1, 0)
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
        if n == 1:
            glColor3f(1, 0.6, 0.2)
        else:
            glColor3f(0.97, 0.97, 0.97)
        glRotatef(-phi, 0.0, 0.0, 1.0)
        glRotatef(cfg.appthetaC + theta, 0, 1, 0)
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
        glRotatef(cfg.appthetaR + theta, 0, 1, 0)
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
        base3 = 135
        painter.setPen(QColor(255, 153, 51))
        if cfg.experiment < 0:
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
        else:
            y = int(0.25 * self.height() + base1 + 5 * step)
            rect = QRect(20, 0, 150, self.height() - y)
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                "Apparatus")
            y = int(0.25 * self.height() + base1 + 4 * step)
            rect = QRect(20, 0, 150, self.height() - y)
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                f"θ (L)eft: {cfg.appthetaL:.1f}°")
            y = int(0.25 * self.height() + base1 + 3 * step)
            rect = QRect(20, 0, 150, self.height() - y)
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                f"θ (C)enter: {cfg.appthetaC:.1f}°")
            y = int(0.25 * self.height() + base1 + 2 * step)
            rect = QRect(20, 0, 150, self.height() - y)
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                f"θ (R)ight: {cfg.appthetaR:.1f}°")
        if self.measurement1:
            painter.setPen(QColor(255, 255, 255))
            half_width = int(self.width() / 2)
            tq_width = int(self.width() * 3 / 2)
            measurements_nb = len(self.measurements1)
            y = int(0.25 * self.height() + base1 + 1 * step)
            rect = QRect(0, y, half_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Measurement: {self.measurement1}")
            y = int(0.25 * self.height() + base1 + 2 * step)
            rect = QRect(0, y, half_width, self.height() - y)
            prob_p1 = np.count_nonzero(
                self.measurements1 == 1) / measurements_nb
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< color 1 > = {prob_p1 * 100:.1f}%")
            y = int(0.25 * self.height() + base1 + 3 * step)
            rect = QRect(0, y, half_width, self.height() - y)
            prob_m1 = np.count_nonzero(
                self.measurements1 == -1) / measurements_nb
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< color 2 > = {prob_m1 * 100:.1f}%")
            # Invert back the results for apparatus 2 if in the config,
            # for correctly displaying the measurement as it would be
            # if the apparatus measure it (so if apparatus 1 shows +1
            # and both are using the same switch, apparatus 2 should
            # show -1 if it is a singlet
            if cfg.invert:
                measurement2_disp = -1 * self.measurement2
            else:
                measurement2_disp = self.measurement2
            y = int(0.25 * self.height() + base1 + 1 * step)
            rect = QRect(0, y, tq_width, self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Measurement: {measurement2_disp}")
            y = int(0.25 * self.height() + base1 + 2 * step)
            rect = QRect(0, y, tq_width, self.height() - y)
            prob_p2 = np.count_nonzero(
                self.measurements2 == 1) / measurements_nb
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< color 1 > = {prob_p2 * 100:.1f}%")
            y = int(0.25 * self.height() + base1 + 3 * step)
            rect = QRect(0, y, tq_width, self.height() - y)
            prob_m2 = np.count_nonzero(
                self.measurements2 == -1) / measurements_nb
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< color 2 > = {prob_m2 * 100:.1f}%")
            same_mask = self.switches1 == self.switches2
            diff_mask = self.switches1 != self.switches2
            num_same = np.sum(same_mask)
            num_diff = np.sum(diff_mask)
            # Count occurrences where measurements have
            # the same value
            equal = np.sum(self.measurements1 == self.measurements2)
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
                             f"{num_same / measurements_nb * 100:.1f}%")
            if num_same > 0:
                y = int(0.25 * self.height() + base1 + 3 * step)
                rect = QRect(0, 0, self.width(), self.height() - y)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                                 "% same results = "
                                 f"{equal_same_mask / num_same * 100:.1f}%")
            y = int(0.25 * self.height() + base2 + 2 * step)
            rect = QRect(0, 0, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "Different Switch")
            y = int(0.25 * self.height() + base2 + 1 * step)
            rect = QRect(0, 0, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "Percentage = "
                             f"{num_diff / measurements_nb * 100:.1f}%")
            if num_diff > 0:
                y = int(0.25 * self.height() + base2 + 0 * step)
                rect = QRect(0, 0, self.width(), self.height() - y)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                                 "% same results = "
                                 f"{equal_diff_mask / num_diff * 100:.1f}%")
            y = int(0.25 * self.height() + base1 + 1 * step)
            rect = QRect(0, y, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Total Measurements: {measurements_nb}")
            y = int(0.25 * self.height() + base1 + 2 * step)
            rect = QRect(0, y, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "% same results = "
                             f"{equal / measurements_nb * 100:.1f}%")
        if cfg.experiment == 2:
            # Compute the probability for Bell's inequality
            c01, p01 = self.calculate_probabilities_exp2(0, 1, 1, -1)
            c12, p12 = self.calculate_probabilities_exp2(1, 2, 1, -1)
            c02, p02 = self.calculate_probabilities_exp2(0, 2, 1, -1)
            if c01 and c12 and c02:
                p1 = (p01 + p12) * 100
                p2 = p02 * 100
                painter.setPen(QColor(255, 153, 51))
                text1 = f"N({cfg.appthetaL}°+,{cfg.appthetaC}°-) + "\
                    f"N({cfg.appthetaC}°+,{cfg.appthetaR}°-) ≥ "\
                    f"N({cfg.appthetaL}°+,{cfg.appthetaR}°-)"
                y = int(0.25 * self.height() + base3 - 4 * step)
                rect = QRect(0, y, self.width(), self.height() - y)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text1)
                text2 = f"1/2*sin^2({cfg.appthetaC - cfg.appthetaL}°) + " \
                    f"1/2*sin^2({cfg.appthetaR - cfg.appthetaC}°)  ≥ " \
                    f"1/2*sin^2({cfg.appthetaR - cfg.appthetaL}°)"
                y = int(0.25 * self.height() + base3 - 3 * step)
                rect = QRect(0, y, self.width(), self.height() - y)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text2)
                text3 = f"{p1:.2f}% ≥ {p2:.2f}%"
                y = int(0.25 * self.height() + base3 - 2 * step)
                rect = QRect(0, y, self.width(), self.height() - y)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text3)
                if p1 < p2:
                    y = int(0.25 * self.height() + base3 - 1 * step)
                    rect = QRect(0, y, self.width(), self.height() - y)
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                                     "Bell's inequality is violated")
            if cfg.verbose:
                if c01:
                    print(f"pass {cfg.appthetaL} and not pass "
                          f"{cfg.appthetaC}: {p01 * 100:.2f}%")
                if c12:
                    print(f"pass {cfg.appthetaC} and not pass "
                          f"{cfg.appthetaR}: {p12 * 100:.2f}%")
                if c02:
                    print(f"pass {cfg.appthetaL} and not pass "
                          f"{cfg.appthetaR}: {p02 * 100:.2f}%")
                if c01 and c12 and c02:
                    print(text1)
                    print(text3)
                    if p1 < p2:
                        print("Bell's inequality is violated")
        painter.end()
        glEnable(GL_DEPTH_TEST)

    def calculate_probabilities_exp2(self, sw_A, sw_B, r_1, r_2):
        cond1 = (self.switches1 == sw_A) & (self.switches2 == sw_B)
        count1 = np.sum(cond1)
        cond2 = (self.switches1 == sw_B) & (self.switches2 == sw_A)
        count2 = np.sum(cond2)
        countnb = count1 + count2
        prob = 0.0
        if count1 > 0:
            prob = np.mean((self.measurements1[cond1] == r_1) & (
                self.measurements2[cond1] == r_2))
        else:
            prob = 0
        if count2 > 0:
            prob2 = np.mean((self.measurements1[cond2] == r_2) & (
                self.measurements2[cond2] == r_1))
            prob = (prob * count1 + prob2 * count2) / countnb
        return countnb, prob

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
            if random_number < 0.5:

                # Simulate measuring the first apparatus
                self.measurement1, self.measurement2 = self.spin.Measure(
                    np.array([self.direction1p[self.button1],
                              self.direction1m[self.button1]]),
                    np.array([self.direction2p[self.button2],
                              self.direction2m[self.button2]]),
                    True)
            else:
                # Simulate measuring the second apparatus
                self.measurement2, self.measurement1 = self.spin.Measure(
                    np.array([self.direction1p[self.button1],
                              self.direction1m[self.button1]]),
                    np.array([self.direction2p[self.button2],
                              self.direction2m[self.button2]]),
                    False)
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
        match cfg.experiment:
            case -1:
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
                desc = f"EPR Experiment - {desc}"
            case 1:
                desc = 'EPR Experiment - Stern-Gerlach magnets with '\
                    'electrons in singlet state '\
                    '| Psi > = 1 / sqrt(2) ( | ud > - | du > )'
            case 2:
                desc = 'EPR Experiment - Polarizers with '\
                    'photons in triplet state '\
                    '| Psi > = 1 / sqrt(2) ( | uu > + | dd > )'

        self.setWindowTitle(desc)

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
        if cfg.experiment < 0:
            self.radioButtonFix.setChecked(True)
        else:
            self.radioButtonRandom.setChecked(True)

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
    parser.add_argument('-e', '--experiment', type=int, choices=[1, 2],
                        help='predefined experiment')
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
    if (args.measurement_number):
        cfg.n = int(args.measurement_number)
    if (args.no_invert):
        cfg.invert = False
    if (args.verbose):
        cfg.verbose = True
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
    if (args.bloch_theta):
        cfg.bloch_t = args.bloch_theta
    if (args.bloch_phi):
        cfg.bloch_p = args.bloch_phi
    # if a specific experiment is selected, set the proper variables
    if args.experiment is not None:
        cfg.experiment = args.experiment
        match cfg.experiment:
            case 1:
                # The detectors are three Stern-Gerlach magnets one
                # oriented along the z axis and the other two in the zx plane
                # with ±120° rotation and the particles are two entangled
                # electrons in the singlet state
                # Particles are in the singlet state
                cfg.stype = 1
                # orientation ±120°
                cfg.appthetaL = 240
                cfg.appthetaC = 0
                cfg.appthetaR = 120
                cfg.theta1 = 0
                cfg.theta2 = 0
                # xz plane
                cfg.phi1 = 0
                cfg.phi2 = 0
                cfg.invert = True
            case 2:
                # The apparatus is composed by two polarizers which
                # send the photons to three photodectors, one oriented
                # along the z axis and the other two in the zx plane
                # with 22.5° and 67.5° rotation and the particles
                # are two entangled photons in the second triplet state
                # Particles are a triplet in the second state
                cfg.stype = 3
                # orientation 22.5° and 67.5°
                cfg.appthetaL = 0
                cfg.appthetaC = 22.5
                cfg.appthetaR = 45
                cfg.theta1 = 0
                cfg.theta2 = 0
                # In the real world, light polarization is typically measured
                # in degrees, and the angle θ can be from 0° to 360°.
                # In the Hilbert space, the angles are typically represented by
                # the state vectors on the Bloch sphere,
                # where θ ranges from 0 to π.
                #  Since vertical and horizontal polarizations are orthogonal
                # and correspond to π/2 in real-world measurements and π
                # on the Bloch sphere, the relationship between the real-world
                # polarization angle `θ_real` and the Hilbert space angle
                # `θ_Hilbert` is given by:
                # θHilbert = 2 * θreal
                cfg.bloch_t = 2
                cfg.bloch_p = 1
                # xz plane
                cfg.phi1 = 0
                cfg.phi2 = 0
                cfg.invert = False
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise RuntimeError('Must be using Python 3')
    main()
