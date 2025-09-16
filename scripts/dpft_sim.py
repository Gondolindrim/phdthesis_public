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
L = 1*10**-3
omega0 = 120*pi

Vinf = 100
P0 = 1000
Q0 = 100

def initial_curr(x):
	Id = x[0]
	Iq = x[1]
	F0 = R*(Id**2 + Iq**2) + Vinf*Id - P0
	F1 = omega0*L*(Id**2 + Iq**2) - Vinf*Iq - Q0
	return [F0,F1]

y = fsolve(initial_curr,[10,1])
print("--> Initial current = Id = {0:12.12f}, Iq={1:12.12f}".format(y[0],y[1]))
I0 = y[0] + 1j*y[1]

# -------------------------------------------------------------------------------------------------
# DEFINING MODEL EQUATIONS
# -------------------------------------------------------------------------------------------------
tsim = np.linspace(0,1, 1*10**4 + 1)

def denom_poly(kI,kP):
	coef2 = L
	coef1 = kP + R - 1j*omega0*L
	coef0 = kI - 1j*(kP + R)*omega0
	return np.polynomial.polynomial.Polynomial([coef0,coef1,coef2])

def muP(kI,kP):
	real = -(kP + R) + np.sqrt((  np.sqrt( ((kP+R)**2 - 4*L*kI - omega0**2*L**2)**2 + 4*omega0**2*L**2*(kP+R)**2) + (kP + R)**2 - 4*L*kI - omega0**2*L**2  )/2);
	imag =  omega0*L + np.sqrt((  np.sqrt( ((kP+R)**2 - 4*L*kI - omega0**2*L**2)**2 + 4*omega0**2*L**2*(kP+R)**2) - (kP + R)**2 + 4*L*kI + omega0**2*L**2  )/2);
	return 1/(2*L)*(real + 1j*imag)

def muN(kI,kP):
	real = -(kP + R) - np.sqrt((  np.sqrt( ((kP+R)**2 - 4*L*kI - omega0**2*L**2)**2 + 4*omega0**2*L**2*(kP+R)**2) + (kP + R)**2 - 4*L*kI - omega0**2*L**2  )/2);
	imag =  omega0*L - np.sqrt((  np.sqrt( ((kP+R)**2 - 4*L*kI - omega0**2*L**2)**2 + 4*omega0**2*L**2*(kP+R)**2) - (kP + R)**2 + 4*L*kI + omega0**2*L**2  )/2);
	return 1/(2*L)*(real + 1j*imag)

#------------------------------------------------ FIRST SCENARIO
print("!!---> FIRST SCENARIO")
deltaI = 0.2*I0

deltaV = 0
Iref = I0 + deltaI
print("--> Setpoint: I*_d = {0:12.12f}, I*_q={1:12.12f}".format(np.real(Iref),np.imag(Iref)))
kP = 0.1
kI = 10
print('--> POLES: mu(-) = {}, mu(+) = {}'.format(muN(kI,kP),muP(kI,kP)))
B = ( (kP*(muP(kI,kP) - 1j*omega0) + kI)*deltaI - (muP(kI,kP) - 1j*omega0)*deltaV)/((muP(kI,kP) - 1j*omega0)*L*(muP(kI,kP) - muN(kI,kP)))
C = ( (kP*(muN(kI,kP) - 1j*omega0) + kI)*deltaI - (muN(kI,kP) - 1j*omega0)*deltaV)/((muN(kI,kP) - 1j*omega0)*L*(muN(kI,kP) - muP(kI,kP)))

print("--> A = {0:12.12f}, B = {1:12.12f}, C = {1:12.12f}".format(deltaI,B,C))

def current_t(t): return Iref + B*cmath.exp(t*(muP(kI,kP) - 1j*omega0)) + C*cmath.exp(t*(muN(kI,kP) - 1j*omega0))

IDP1 = [ current_t(tsim[k]) for k in range(len(tsim))]

fig1, [ax1, ax2] = pyplot.subplots(2, 1)
ax1.plot(tsim,np.real(IDP1), color='blue', linewidth=1)
ax2.plot(tsim,np.imag(IDP1), color='red', linewidth=1)
ax1.axhline(y=np.real(Iref), linestyle='--', color='black', linewidth=0.75)
ax2.axhline(y=np.imag(Iref), linestyle='--', color='black', linewidth=0.75)

fig2, ax3 = pyplot.subplots(1, 1)
abscurr = [abs(IDP1[k]) for k in range(len(IDP1))]
argcurr = [phase(IDP1[k]) for k in range(len(IDP1))]
itime1 = [ abscurr[k]*cos(omega0*tsim[k] + argcurr[k]) for k in range(len(tsim))]
ax3.plot(tsim,itime1 , color='blue', linewidth=1)

#------------------------------------------------ SECOND SCENARIO

print("!!---> SECOND SCENARIO")
deltaI = 0
deltaV = 0.05*Vinf
Iref = I0 + deltaI
print("--> Setpoint: I*_d = {0:12.12f}, I*_q={1:12.12f}".format(np.real(Iref),np.imag(Iref)))
kP = 0.1
kI = 10
B = ( (kP*(muP(kI,kP) - 1j*omega0) + kI)*deltaI - (muP(kI,kP) - 1j*omega0)*deltaV)/((muP(kI,kP) - 1j*omega0)*L*(muP(kI,kP) - muN(kI,kP)))
C = ( (kP*(muN(kI,kP) - 1j*omega0) + kI)*deltaI - (muN(kI,kP) - 1j*omega0)*deltaV)/((muN(kI,kP) - 1j*omega0)*L*(muN(kI,kP) - muP(kI,kP)))

print("--> A = {0:12.12f}, B = {1:12.12f}, C = {1:12.12f}".format(deltaI,B,C))

def current_t(t): return Iref + B*cmath.exp(t*(muP(kI,kP) - 1j*omega0)) + C*cmath.exp(t*(muN(kI,kP) - 1j*omega0))

IDP2 = [ current_t(tsim[k]) for k in range(len(tsim))]

fig3, [ax4, ax5] = pyplot.subplots(2, 1)
ax4.plot(tsim,np.real(IDP2), color='blue', linewidth=1)
ax5.plot(tsim,np.imag(IDP2), color='red', linewidth=1)
ax4.axhline(y=np.real(Iref), linestyle='--', color='black', linewidth=0.75)
ax5.axhline(y=np.imag(Iref), linestyle='--', color='black', linewidth=0.75)

fig4, ax6 = pyplot.subplots(1, 1)
abscurr = [abs(IDP2[k]) for k in range(len(IDP2))]
argcurr = [phase(IDP2[k]) for k in range(len(IDP2))]
itime2 = [ abscurr[k]*cos(omega0*tsim[k] + argcurr[k]) for k in range(len(tsim))]
ax6.plot(tsim,itime2 , color='blue', linewidth=1)

fig5, ax7 = pyplot.subplots(1, 1)
ax7.plot(np.real(IDP1),np.imag(IDP1))
fig6, ax8 = pyplot.subplots(1, 1)
ax8.plot(np.real(IDP2),np.imag(IDP2))

dIDP1 = np.gradient(IDP1,tsim)
#vel1 = np.abs([ np.real(IDP1[k]*np.conjugate(dIDP1[k]))/abs(IDP1[k]) for k in range(len(tsim))])
vel1 = np.abs(dIDP1)

dIDP2 = np.gradient(IDP2,tsim)
#vel2 = np.abs([ np.real(IDP2[k]*np.conjugate(dIDP2[k]))/abs(IDP2[k]) for k in range(len(tsim))])
vel2 = np.abs(dIDP2)

data1 = np.array([tsim,np.real(IDP1),np.imag(IDP1),itime1,vel1]).T
np.savetxt("data_dpft_sim_scenario1.csv", data1, delimiter=",")

data2 = np.array([tsim,np.real(IDP2),np.imag(IDP2),itime2,vel2]).T
np.savetxt("data_dpft_sim_scenario2.csv", data2, delimiter=",")

pyplot.show()
