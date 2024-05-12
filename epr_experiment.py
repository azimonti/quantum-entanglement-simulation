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
from mod_spin_operators import TwolSpins
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QPushButton, QLabel
from PyQt6.QtWidgets import QButtonGroup, QRadioButton
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QFont, QColor
from PyQt6.QtCore import QRect
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

cfg = SimpleNamespace(stype=1, n=100, color_up=[0, 1, 0], color_down=[1, 0, 0])

description = (
    'This script simulates two entangled spins following '
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
    '"-n, --measurement_number" (default = 100).\n\n'
    'It is possible to set the color for the spin up (| +1 >) result '
    'with the command line option "-u, --color_up" (default = green) '
    'and for the spin down (| -1 >) with "-d, --color_down (default = red).'
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

        # Set the Spin Type
        spin = TwolSpins()
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
        glTranslatef(-2.5, 0.0, 0.0)  # Move to the left
        self.drawApparatus(True)
        glTranslatef(5.0, 0.0, 0.0)  # Move to the right
        self.drawApparatus(False)
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
        self.drawArrows(button)

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
        glTranslatef(-1.05, -0.1, -0.1)
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

    def drawArrows(self, n):
        glPushMatrix()
        if n == 1:
            glColor3f(1, 0.6, 0.2)
        else:
            glColor3f(0.97, 0.97, 0.97)
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
        glRotatef(240, 0, 1, 0)
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
        glRotatef(120, 0, 1, 0)
        glBegin(GL_LINES)
        glVertex3f(0., 0, 0)
        glVertex3f(0., 0, 2.2)
        glVertex3f(0., 0, 2.19)
        glVertex3f(-0.2, 0, 1.9)
        glVertex3f(0., 0, 2.19)
        glVertex3f(0.2, 0, 1.9)
        glEnd()
        glPopMatrix()

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def measure(self, measure_single: bool):
        # Randomize the button is selected
        if not self.isFixed:
            self.button1 = random.randint(0, 2)
            self.button2 = random.randint(0, 2)

        random_number = random.uniform(0, 1)
        self.measurement1 = 1 if random_number < 0.5 else -1
        random_number = random.uniform(0, 1)
        self.measurement2 = 1 if random_number < 0.5 else -1
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
        self.opengl_widget.measure(True)

    def on_button2_clicked(self):
        self.opengl_widget.measure(False)


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
    parser.add_argument('-n', '--measurement_number', type=int, default=100,
                        help='Number of simultaneous measurements - '
                        'Default: 100')
    parser.add_argument('-u', '--color_up', type=parse_color,
                        default=(0.0, 1.0, 0.0),
                        help='Set the spin up (| +1 >) color as '
                        'comma-separated RGB values (0-255). '
                        'Example: -c 0,255,0 - Default: red')
    parser.add_argument('-d', '--color_down', type=parse_color,
                        default=(1.0, 0.0, 0.0),
                        help='Set the spin down (| -1 >) color as '
                        'comma-separated RGB values (0-255). '
                        'Example: -c 255,0,0 - Default: red')

    args = parser.parse_args()
    if (args.simul_type):
        cfg.stype = int(args.simul_type)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    main()
