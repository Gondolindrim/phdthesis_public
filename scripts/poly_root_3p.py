# -------------------------------------------------------------------------------------------------
# IMPORTING LIBRARIES
# -------------------------------------------------------------------------------------------------
# Python standard math/scientific libraries
import numpy as np

# Defining constants
pi = np.pi
exp = np.exp
cos = np.cos
sin = np.sin

# Defining math functions
inv = np.linalg.inv
transpose = np.transpose
array = np.array

# FSOLVE for nonlinear equation solving
import scipy
from scipy.optimize import fsolve as fsolve
import scipy.integrate as spi

from cmath import phase as phase

# MatplotLib and PyPlot for plotting
import matplotlib.pyplot as pyplot
import matplotlib as mplot
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('text.latex', preamble=r' \usepackage{amsmath} ')

# Time library
import time

import cmath

# -------------------------------------------------------------------------------------------------
# PROBLEM PARAMETERS
# -------------------------------------------------------------------------------------------------
R = 1*10**-2
L = 1*10**0
omega0 = 200

# -------------------------------------------------------------------------------------------------
# DEFINING MODEL EQUATIONS
# -------------------------------------------------------------------------------------------------
def Pk(k): return sum([omega0**i for i in range(k+1)])
print(Pk(0))

j = complex(0,1)
s = 1000
P = np.polynomial.polynomial.Polynomial(np.polynomial.polynomial.polyfromroots((s*(-5 + 1j*2), s*(-1.5 + 1j*1), s*(-1), s*(-3 + 1j*3), s*(-3 - 1j*2), s*(-2 + 1j*2))))

fig1, ax1 = pyplot.subplots(1, 1)

for i in range(6): ax1.plot(np.real(P.deriv(i).roots()), np.imag(P.deriv(i).roots()), 'o')

polyres = np.polynomial.polynomial.Polynomial((0))
for i in range(6): polyres = polyres + Pk(i)*P.deriv(i)

devsum = np.polynomial.polynomial.Polynomial((0))
for i in range(6): devsum = devsum + P.deriv(i)


print(polyres)
#tP = polyres
#totalP = np.polynomial.polynomial.Polynomial(tP)
print(polyres.roots())
#
ax1.plot(np.real(polyres.roots()), np.imag(polyres.roots()), '*', color='black')
ax1.plot(np.real(devsum.roots()), np.imag(devsum.roots()), 'o', color='black')
for i in range(6): ax1.plot(np.real(polyres.deriv(i).roots()), np.imag(polyres.deriv(i).roots()), '*', color='black')

pyplot.show()
