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
L = 4*10**-3
C = 1*10**-3
R = 1*10**2

# -------------------------------------------------------------------------------------------------
# DEFINING MODEL EQUATIONS
# -------------------------------------------------------------------------------------------------
def mv(t): return 10

def mvdot(t): return 0

# omega is of the form
# omega0*( 1 + M*exp(-alpha * t)*sin(beta *t))
M = 0.1
alpha = -5
beta = 10*pi
omega0 = 120*pi

print(((1/(L*C))**0.5)/omega0)
print(((1/(R*C)))/omega0)
def omega(t):	return omega0*(1 + M*exp(alpha*t)*sin(beta*t))
def dot_omega(t): return omega0*M*(alpha*exp(alpha*t)*sin(beta*t) + beta*exp(alpha*t)*cos(beta*t))
def psi(t): return omega0*(t + M*(beta + exp(alpha*t)*(alpha*sin(beta*t) - beta*cos(beta*t)))/(alpha**2 + beta**2))

def complex_diff_eq(x, t):
	# x[0] = Re(VR)
	# x[1] = d(Re(VR))
	# x[2] = Im(VR)
	# x[3] = d(Im(VR)))
	f0 = x[1] # Derivative of Re(VR)
	f1 = -( - mv(t)*cos(psi(t)- omega0*t)/(L*C) + x[1]*1/(R*C) - 2*omega0*x[3] + x[0]*( 1/(L*C) - omega0**2 ) - x[2]*(1/(R*C)*omega0) ) # Derivative of d(Re(VR))
	f2 = x[3]
	f3 = -( - mv(t)*sin(psi(t) - omega0*t)/(L*C) + x[3]*1/(R*C) + 2*omega0*x[1] + x[2]*( 1/(L*C) - omega0**2 ) + x[0]*(1/(R*C)*omega0) ) # Derivative of d(Re(VR))
	return np.array([f0, f1, f2, f3])

def complex_diff_eq_init(x): 
	vR0 = mv(0)*cos(psi(0))
	vI0 = mv(0)*sin(psi(0))
	f0 = -( - mv(0)/(L*C) + x[0]*1/(R*C) - 2*omega0*x[1] + vR0*( 1/(L*C) - omega0**2) - vI0*(1/(R*C)*omega0) ) # Derivative of d(Re(VR))
	f1 = -(                 x[1]*1/(R*C) + 2*omega0*x[0] + vI0*( 1/(L*C) - omega0**2) + vR0*(1/(R*C)*omega0) ) # Derivative of d(Re(VR))
	return np.array([f0, f1])

# Calculating the initial conditions for d(Re(VR)) and d(Im(VR)) is needed.
# In principle, the system is such that vR(0) = v(0) = mv(0) because in initial time the inductor is a short and the capacitor is an open circuit.
# Supposing d(v(0)) = 0, this gives the needed initial conditions for the time DE.
# In the complex DE, however, Re(VR)(0) = mv(0) and Im(VR)(0) = 0. The derivatives of the real and imaginary parts are not null however, and they
#	must be adjusted so that their reconstruction P_D(-1)(VR) matches vR at time zero.
#	It must be noted that the initial condition is just that, an initial condition; at infinity, all transients will fade and 
#	PD(-1)(VR) will tend to vR(t) nonetheless.
def time_diff_eq(x, t):
	# x[0] = vR
	# x[1] = dVR
	f0 = x[1] # Derivative of Re(VR)
	f1 = 1/(L*C)*mv(t)*cos(psi(t)) - 1/(L*C)*x[0] - 1/(R*C)*x[1] # Derivative of d(Re(VR))
	return np.array([f0, f1])

t = np.linspace(0, 1, 1*10**3 + 1)
initial_conds = fsolve(complex_diff_eq_init, [0,0]) # Calculating the initial value of the derivatives of Re(VR) and Im(VR)
print(complex_diff_eq_init(initial_conds))
y = spi.odeint(complex_diff_eq, [mv(0),initial_conds[0],0,initial_conds[1]], t)
ytime = spi.odeint(time_diff_eq, [mv(0)*cos(psi(0)),0] , t)

fig1, [ax1, ax2] = pyplot.subplots(2, 1)
fig2, [ax3, ax4] = pyplot.subplots(2, 1)
fig3, [ax5, ax6] = pyplot.subplots(2, 1)
fig4, [ax7, ax8] = pyplot.subplots(2, 1)

v_signal = np.array([abs(y[k,0] + 1j*y[k,2])*cos(omega0*t[k] + phase(y[k,0] + 1j*y[k,2])) for k in range(len(t))])

ydiffeomorphic = np.array([ np.exp(-1j*(psi(t[k]) - omega0*t[k]))*(y[k,0] + 1j*y[k,2]) for k in range(len(t))])

mvr  = np.array([abs(y[k,0] + 1j*y[k,2]) for k in range(len(t))])
phasevr = np.array([phase(y[k,0] + 1j*y[k,2]) for k in range(len(t))])

ax1.plot(t,mvr,linewidth=0.5 ,color="red",label="$m$")
ax2.plot(t,phasevr,linewidth=0.5 ,color="blue",label="$\phi$")

ax3.plot(t,v_signal,linewidth=0.5 ,color="red",label="$P_D^{-1}(V_R(t))$")
ax3.plot(t,ytime[:,0],linewidth=0.5 ,color="blue",label="$v(t)$")
ax3.plot(t, mvr, linewidth=0.5,color="green",label="$\pm m_{v_R}(t)$")
ax3.plot(t,-1*np.array(mvr), linewidth=0.5,color="green",)

error = np.array([((v_signal[k] - ytime[k,0])/abs(y[k,0] + 1j*y[k,2])) for k in range(len(t))])
ax4.plot(t,100*error,linewidth=0.5 ,color="red",label="$V_o$")

ic = C*np.array([(y[k,1] + 1j*y[k,3]) + 1j*omega(t[k])*(y[k,0] + 1j*y[k,2]) for k in range(len(t))])
i = ic + y[:,0] + 1j*y[:,2]/R # Current through supply

ax5.plot(t,y[:,0],linewidth=0.5,color="red",label="$V_{R0}$")
ax6.plot(t,y[:,2],linewidth=0.5,color="red",label="$V_{R0}$")

ax5.plot(t,[np.real(ydiffeomorphic[k]) for k in range(len(t))],linewidth=0.5,color="blue",label="$V_{R}$")
ax6.plot(t,[np.imag(ydiffeomorphic[k]) for k in range(len(t))],linewidth=0.5,color="blue",label="$V_{R}$")

ax1.set_xlabel(r'Time (s)')
ax1.set_ylabel(r'$\left| V_R(t)\right|$ (V)')
ax1.grid(True)
ax2.set_xlabel(r'Time (s)')
ax2.set_ylabel(r'arg$\left(V_R(t)\right)$ (rad) ')
ax2.grid(True)
ax3.set_xlabel(r'Time (s)')
ax3.set_ylabel(r'Time signals')
ax3.grid(True)
ax4.set_xlabel(r'Time (s)')
ax4.set_ylabel(r'Relative error $100\times\left[\dfrac{v(t) - P_D^{-1}\left(V_R(t)\right)}{m_v(t)}\right]$')
ax4.grid(True)

ax1.legend(loc="upper left")
ax2.legend(loc="upper left")
ax3.legend(loc="upper left")
ax4.legend(loc="upper left")
ax5.legend(loc="upper left")
ax6.legend(loc="upper left")

data = np.array([t,ytime[:,0],np.real(ydiffeomorphic),np.imag(ydiffeomorphic),v_signal,y[:,0],y[:,2],error]).T

ax7.plot(t,np.abs(ydiffeomorphic))
ax8.plot(t,np.angle(ydiffeomorphic))

np.savetxt("data_omega0.csv", data, delimiter=",")
pyplot.show()
