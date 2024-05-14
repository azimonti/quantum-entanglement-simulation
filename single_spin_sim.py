#!/usr/bin/env python3
'''
/************************/
/*  single_spin_sim.py  */
/*    Version 1.0       */
/*     2024/05/11       */
/************************/
'''
import argparse
import cmath
import math
from mod_spin_operators import SingleSpin
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

cfg = SimpleNamespace(stype=1, color=(1, 1, 1))

description = (
    'This script simulates a single spin following quantum '
    'mechanics principles.\n'
    'Simulation types available (-t SIMUL_TYPE, --simul_type SIMUL_TYPE):\n'
    '0 - No Time evolution\n'
    '1 - The spin is prepared to be always in the up direction'
    '(reset at each time step) [DEFAULT]\n'
    '2 - The spin is prepared to be always in the left direction'
    '(reset at each time step)\n'
    '3 - The spin is prepared to be always in the inner direction'
    '(reset at each time step)\n'
    '4 - The spin is collapsed in the direction of the measurement\n\n'
    'It is possible to set the color for apparatus with the '
    'command line option "-c, --color" (default = white).'
)


class SimulationThread(QThread):
    # Currently is not used but it is ready in case there is a time
    # evolution for example updating with a magnetic field applied
    result = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.current_state = None

    def run(self):
        simul_spin = SingleSpin()

        # initial condition for the case needed
        match cfg.stype:
            case 1:
                self.current_state = simul_spin.u
            case 2:
                self.current_state = simul_spin.l
            case 3:
                self.current_state = simul_spin.i
            case 4:
                self.current_state = simul_spin.u
            case _:
                raise ValueError(
                    f"Incorrect simulation type {cfg.stype}")
        while True:
            self.result.emit(self.current_state)
            time.sleep(1)  # slows down the loop for demonstration purposes

    def collapse_wave_function(self, state: np.ndarray):
        self.current_state = state


class OpenGLWidget(QOpenGLWidget):

    def __init__(self, parent):
        super(OpenGLWidget, self).__init__(parent)
        self.a_theta = 0
        self.a_phi = 0
        self.measurement = None
        self.count_p1 = 0
        self.count_m1 = 0
        self.current_state = None
        self.num_measurements = 0
        self.spin = SingleSpin()

    @property
    def apparatus_direction(self):
        return np.array([
            [np.cos(self.a_theta / 2)],
            [np.exp(1j * self.a_phi) * np.sin(self.a_theta / 2)]])

    @property
    def apparatus_opposite_direction(self):
        return np.array([
            [np.cos(np.pi / 2 + self.a_theta / 2)],
            [np.exp(1j * self.a_phi) * np.sin(
                np.pi / 2 + self.a_theta / 2)]])

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glColor3f(*cfg.color)
        # Adjust the camera view
        gluLookAt(0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        # Rotate the rectangle along Y-axis
        a_theta_deg = math.degrees(self.a_theta)
        a_phi_deg = math.degrees(self.a_phi)
        # print(f"apparatus theta = {a_theta_deg}\n phi = {a_phi_deg}")

        # Apply the rotation
        glRotatef(-a_phi_deg, 0.0, 0.0, 1.0)
        glRotatef(a_theta_deg, 0.0, 1.0, 0.0)
        # Draw a rectangle on ZX plane
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
        glPopMatrix()

        if self.measurement is not None:
            painter = QPainter(self)
            painter.setFont(QFont('Arial', 14))
            painter.setPen(QColor(255, 255, 255))
            y = int(0.25 * self.height() + 20)
            rect = QRect(0, y, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Measurement: {self.measurement}")
            y = int(0.25 * self.height() + 55)
            rect = QRect(0, y, self.width(), self.height() - y)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"Total Measurements: {self.num_measurements}")
            y = int(0.25 * self.height() + 90)
            rect = QRect(0, y, self.width(), self.height() - y)
            prob_p1 = self.count_p1 / self.num_measurements * 100
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< +1 > = {prob_p1:.1f}%")
            y = int(0.25 * self.height() + 125)
            rect = QRect(0, y, self.width(), self.height() - y)
            prob_m1 = self.count_m1 / self.num_measurements * 100
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"< -1 > = {prob_m1:.1f}%")

            y = int(0.75 * self.height() - 40)
            x = int(0.25 * self.width())
            rect = QRect(x, 0, self.width() - x, y)
            painter.setPen(QColor(255, 255, 0))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "current state")
            # display the current spin
            glPushMatrix()
            glMatrixMode(GL_MODELVIEW)
            glTranslatef(1.0, 0., 0.7)
            glColor3f(1.0, 1.0, 0.0)

            # compute the spin angle
            s_theta = 2 * np.arctan2(
                abs(self.current_state[1][0]), abs(self.current_state[0][0]))
            if self.current_state[1][0] != 0:
                s_phi = cmath.phase(self.current_state[1][0])
            else:
                s_phi = 0
            # If the real part is negative, set it > pi
            if self.current_state[0][0].real < 0:
                s_theta = 2 * np.pi - s_theta

            if s_phi < 0:
                s_phi += 2 * np.pi
            # Convert theta and phi to degrees
            s_theta_deg = math.degrees(s_theta)
            s_phi_deg = math.degrees(s_phi)
            # print(f"spin theta = {s_theta_deg}\nspin phi = {s_phi_deg}")

            y = int(0.25 * self.height() + 160)
            rect = QRect(0, y, self.width(), self.height() - y)
            prob_m1 = self.count_m1 / self.num_measurements

            # calculate the dot product
            dot_product = np.sin(self.a_theta) * np.sin(s_theta) * \
                np.cos(self.a_phi - s_phi) + \
                np.cos(self.a_theta) * np.cos(s_theta)

            # calculate the angle difference
            angle_difference = np.arccos(dot_product)
            # convert angle difference from radians to degrees
            angle_difference_degrees = np.degrees(angle_difference)
            cos_half_alpha_2 = np.cos(
                np.deg2rad(angle_difference_degrees / 2))**2 * 100
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"cos(θ_mn/2)^2(in %): {cos_half_alpha_2:.1f}%")
            painter.end()

            # Apply the rotation - Order is important
            glRotatef(s_phi_deg, 0., 0., 1.)
            glRotatef(s_theta_deg, 0., 1., 0.)

            # Draw an arrow
            glBegin(GL_LINES)
            glVertex3f(0., 0, -0.1)
            glVertex3f(0., 0, 0.1)
            glVertex3f(0., 0, 0.1)
            glVertex3f(-0.1, 0, 0.)
            glVertex3f(0., 0, 0.1)
            glVertex3f(0.1, 0, 0.)
            glEnd()
            glPopMatrix()

        glFlush()

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def update_rotation_theta(self, value: float):
        self.a_theta = np.deg2rad(value)
        self.update()

    def update_rotation_phi(self, value: float):
        self.a_phi = np.deg2rad(value)
        self.update()

    def measure(self, current_state: np.ndarray):
        self.current_state = current_state
        # get the measurement direction
        direction = np.array([
            np.cos(self.a_theta / 2),
            np.exp(1j * self.a_phi) * np.sin(self.a_theta / 2)])
        prob_p1 = np.abs(np.vdot(direction, self.current_state)) ** 2
        # Generate a random number between 0 and 1
        random_number = random.uniform(0, 1)

        # Perform the measurement in apparatus direction
        self.num_measurements += 1
        if random_number < prob_p1:
            self.count_p1 += 1
            self.measurement = 1
        else:
            self.count_m1 += 1
            self.measurement = -1
        self.update()


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
                desc = 'spin always |up>'
            case 2:
                desc = 'spin always |left>'
            case 3:
                desc = 'spin always |inner>'
            case 4:
                desc = 'spin takes measurement direction'
        self.setWindowTitle(f"Single quantum spin simulation: {desc}")

        self.opengl_widget = OpenGLWidget(self)

        self.slider1 = QSlider(Qt.Orientation.Horizontal, self)
        self.slider1.setRange(0, 360)
        self.slider1.valueChanged.connect(self.update_rotation_theta)
        self.label1 = QLabel("Apparatus Rotation θ: 0", self)

        self.slider2 = QSlider(Qt.Orientation.Horizontal, self)
        self.slider2.setRange(0, 360)
        self.slider2.valueChanged.connect(self.update_rotation_phi)
        self.label2 = QLabel("Apparatus Rotation 𝜙: 0", self)

        self.button = QPushButton('Make measurement', self)
        self.button.clicked.connect(self.on_button_clicked)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.opengl_widget)
        self.gridlayout = QGridLayout()
        self.gridlayout.addWidget(self.slider1, 0, 0)
        self.gridlayout.addWidget(self.label1, 0, 1)
        self.gridlayout.addWidget(self.slider2, 1, 0)
        self.gridlayout.addWidget(self.label2, 1, 1)
        self.gridlayout.addWidget(self.button, 2, 0, 1, 2)
        self.layout.addLayout(self.gridlayout)

    def store_simul_spin(self, current_simul_state: np.ndarray):
        self.current_state = current_simul_state

    def update_rotation_theta(self, value: float):
        self.opengl_widget.update_rotation_theta(value)
        self.label1.setText(f"Apparatus Rotation θ: {value}")

    def update_rotation_phi(self, value: float):
        self.opengl_widget.update_rotation_phi(value)
        self.label2.setText(f"Apparatus Rotation 𝜙: {value}")

    def on_button_clicked(self):
        self.opengl_widget.measure(self.current_state)
        match cfg.stype:
            case 1 | 2 | 3:
                pass
            case 4:
                if self.opengl_widget.measurement == 1:
                    state = self.opengl_widget.apparatus_direction
                else:
                    state = self.opengl_widget.apparatus_opposite_direction
                self.simulation_thread.collapse_wave_function(state)


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
    seed_value = 5692
    random.seed(seed_value)
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=CustomHelpFormatter)
    parser.add_argument('-t', '--simul_type', help='simulation type',
                        required=False)
    parser.add_argument('-c', '--color', type=parse_color,
                        help='Set the apparatus color as comma-separated '
                        'RGB values (0-255). Example: -c 255,0,0')
    args = parser.parse_args()
    if (args.simul_type):
        cfg.stype = int(args.simul_type)
    if (args.color):
        cfg.color = args.color
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    main()
