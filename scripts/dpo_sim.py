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

# -------------------------------------------------------------------------------------------------
# PROBLEM PARAMETERS
# -------------------------------------------------------------------------------------------------
R1 = 1*10**3
R2 = 2*10**3
CF = 1*10**-5
RF = 1*10**3

# -------------------------------------------------------------------------------------------------
# DEFINING MODEL EQUATIONS
# -------------------------------------------------------------------------------------------------
# omega is of the form
# omega0*( 1 + M*exp(-alpha * t)*sin(beta *t))
M1 = 2
alpha1 = -5
beta1 = 10*pi
m = 10

M2 = 1
alpha2 = -10
beta2 = 20*pi

omega0 = 120*pi

def omega(t,M,alpha,beta): return omega0*(1 + M*exp(alpha*t)*sin(beta*t))
def psi(t,M,alpha,beta): return omega0*(t + M*(beta + exp(alpha*t)*(alpha*sin(beta*t) - beta*cos(beta*t)))/(alpha**2 + beta**2))
def deltapsi(t,M,alpha,beta): return omega0*M*(beta + exp(alpha*t)*(alpha*sin(beta*t) - beta*cos(beta*t)))/(alpha**2 + beta**2)

def diffeq_v1(x, t):
	# x[0] = realvo1(t)
	# x[1] = imagvo1(t)
	f0 = 1/CF* ( -m*(1 + M1*exp(alpha1*t)*sin(beta1*t))*cos(0)/R1 - (-CF*omega0*x[1] + 1/RF*x[0]) ) 
	f1 = 1/CF* ( -m*(1 + M1*exp(alpha1*t)*sin(beta1*t))*sin(0)/R1 - ( CF*omega0*x[0] + 1/RF*x[1]) ) 
	return np.array([f0,f1])

def diffeq_v2(x, t):
	# x[0] = realvo1(t)
	# x[1] = imagvo1(t)
	f0 = 1/CF* ( -m*cos(deltapsi(t,M2,alpha2,beta2))/R2 - (-CF*omega0*x[1] + 1/RF*x[0]) ) 
	f1 = 1/CF* ( -m*sin(deltapsi(t,M2,alpha2,beta2))/R2 - ( CF*omega0*x[0] + 1/RF*x[1]) ) 
	return np.array([f0,f1])

def diffeq_time_v1(x,t): return 1/CF* ( -m*(1 + M1*exp(alpha1*t)*sin(beta1*t))*cos(omega0*t)/R1 - 1/RF*x)
def diffeq_time_v2(x,t): return 1/CF* ( -m*cos(psi(t,M2,alpha2,beta2))/R2 - 1/RF*x)

t = np.linspace(0, 1, 1*10**4 + 1)
v10 = -m/R1 * 1/(1j*omega0*CF + 1/RF)
v20 = -m/R2 * 1/(1j*omega0*CF + 1/RF)
y1 = spi.odeint(diffeq_v1, [np.real(v10),np.imag(v10)], t)
y2 = spi.odeint(diffeq_v2, [np.real(v20),np.imag(v20)], t)

V1DP = [y1[k,0] + 1j*y1[k,1] for k in range(len(t))]
V2DP = [y2[k,0] + 1j*y2[k,1] for k in range(len(t))]
VODP = [y1[k,0] + y2[k,0] + 1j*(y1[k,1] + y2[k,1]) for k in range(len(t))]

y1 = spi.odeint(diffeq_time_v1, [abs(v10)*cos(phase(v10))], t)
y2 = spi.odeint(diffeq_time_v2, [abs(v20)*cos(phase(v20))], t)

v1dp_t = np.array([abs(V1DP[k])*cos(omega0*t[k] + phase(V1DP[k])) for k in range(len(t))])
v2dp_t = np.array([abs(V2DP[k])*cos(omega0*t[k] + phase(V2DP[k])) for k in range(len(t))])
vodp_t = np.array([abs(VODP[k])*cos(omega0*t[k] + phase(VODP[k])) for k in range(len(t))])

fig1, [ax1, ax2] = pyplot.subplots(2, 1)
ax1.plot(t,np.real(V1DP), color='blue', linewidth = 1, label=r'$V_{od}^1$')
ax2.plot(t,np.imag(V1DP), color='blue', linewidth = 1, label=r'$V_{oq}^1$')

ax1.plot(t,np.real(V2DP), color='red', linewidth = 1, label=r'$V_{od}^2$')
ax2.plot(t,np.imag(V2DP), color='red', linewidth = 1, label=r'$V_{oq}^2$')

fig2, [ax3, ax4] = pyplot.subplots(2, 1)
ax3.plot(t,y1[:,0], color='blue', linewidth = 1, label=r'$v_{o}^1$')
ax3.plot(t,v1dp_t, color='red', linewidth = 1, label=r'$v_{1}^{DP}$')

ax4.plot(t,y2[:,0], color='blue', linewidth = 1, label=r'$v_{o}^2$')
ax4.plot(t,v2dp_t, color='red', linewidth = 1, label=r'$v_{2}^{DP}$')

fig3, ax5 = pyplot.subplots(1, 1)
ax5.plot(t,vodp_t, color='blue', linewidth = 1, label=r'$v_{o}^{DP}$')
ax5.plot(t,y1[:,0] + y2[:,0], color='red', linewidth = 1, label=r'$v_{o}^{T}$')

ax1.legend(loc="upper left")
ax2.legend(loc="upper left")
ax3.legend(loc="upper left")
ax4.legend(loc="upper left")
ax5.legend(loc="upper left")

data = np.array([t,np.real(V1DP),np.imag(V1DP),np.real(V2DP),np.imag(V2DP),vodp_t,y1[:,0]+y2[:,0]]).T
np.savetxt("data_dpo_sim.csv", data, delimiter=",")

pyplot.show()
