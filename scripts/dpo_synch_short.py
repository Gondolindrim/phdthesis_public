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
r = 0.01
X = 0.1
x = 0.5
RL = 2.5

Vinf = 1.1
arginf = 3*pi/180
omega0 = 120*pi
delta0 = 0
H = 1
D = 0

to = 0.1 # 0.14 seems to be the estimate of COT, 0.135 already shows big delay
P0 = 1
Q0 = 0.1

kp = 10

# -------------------------------------------------------------------------------------------------
# DEFINING MODEL EQUATIONS
# -------------------------------------------------------------------------------------------------
omega0 = 120*pi

F = np.zeros(4)

# Calculating initial conditions. We assume that the machine QD frame is initially in phase with the grid QD frame,
# such that delta(0) = 0
def initial_machine_noshort(z):
	Id = z[0]
	Iq = z[1]
	Ed = z[2]
	Eq = z[3]

	V = (Ed + 1j*Eq) - (r + 1j*x)*(Id + 1j*Iq)
	S = (np.real(V) + 1j*np.imag(V))*(Id - 1j*Iq)

	F[0] = np.real(S) - P0
	F[1] = np.imag(S) - Q0

	expr = (RL/(X*x) + 1j*2/x)*(Ed + 1j*Eq)*np.exp(1j*delta0) - RL/(x*X)*Vinf*np.exp(1j*arginf) - (r*RL/(X*x) - 1 + 1j*( (RL + r)/x + RL/X))*(Id + 1j*Iq)
	F[2] = np.real(expr)
	F[3] = np.imag(expr)
	return F

initial_conds = fsolve(initial_machine_noshort, [0.4156,0.445,1,0.5])
print(initial_conds)
print(initial_machine_noshort(initial_conds))

Id0 = initial_conds[0]
Iq0 = initial_conds[1]
Ed0 = initial_conds[2]
Eq0 = initial_conds[3]

V0 = (Ed0 + 1j*Eq0) - (r + 1j*x)*(Id0 + 1j*Iq0)
loadpower = np.abs(V0)**2/RL

PM0 = Ed0*Id0 + Eq0*Iq0

print('--> Initial conditions: EPD0 = {}, EPQ0 = {}, ID0 = {}, IQ0 = {}, PM0 = {}, Power to load = {}'.format(Ed0, Eq0, Id0, Iq0, PM0, loadpower))

def machine_short(z,t):
	omega = z[0]
	delta = z[1]
	Id = z[2]
	Iq = z[3]
	
	Pe = np.real((Ed0 + 1j*Eq0)*np.exp(1j*omega0*delta)*(Id - 1j*Iq))

	PM = PM0 - kp*omega

	F0 = (PM - Pe - D*omega)/(2*H)
	F1 = omega

	expr = omega0/x*( (Ed0 + 1j*Eq0)*np.exp(1j*omega0*delta) - (r + 1j*x)*(Id + 1j*Iq) )
	F2 = np.real(expr)
	F3 = np.imag(expr)

	return [F0,F1,F2,F3]

def machine_short_quasi(z,t):
	omega = z[0]
	delta = z[1]
	expr = ((Ed0 + 1j*Eq0)*np.exp(1j*omega0*delta))/(r + 1j*x)
	Id = np.real(expr)
	Iq = np.imag(expr)
	Pe = np.real((Ed0 + 1j*Eq0)*np.exp(1j*omega0*delta)*(Id - 1j*Iq))

	PM = PM0 - kp*omega

	F0 = (PM - Pe - D*omega)/(2*H)
	F1 = omega

	return [F0,F1]

def machine_noshort(z,t):
	omega = z[0]
	delta = z[1]
	Id = z[2]
	Iq = z[3]

	dId = z[4]
	dIq = z[5]

	Pe = np.real((Ed0 + 1j*Eq0)*np.exp(1j*omega0*delta)*(Id - 1j*Iq))

	PM = PM0 - kp*omega
	F0 = (PM - Pe - D*omega)/(2*H)
	F1 = omega

	expr = omega0**2 * (\
		-( (RL + r)/x + RL/X + 2*1j)*1/omega0*(dId + 1j*dIq) + \
		-( r*RL/(x*X) - 1 + 1j*( (RL + r)/x + RL/X))*(Id + 1j*Iq) + \
		+( omega/x + RL/(x*X) + 1j*2/x)*(Ed0 + 1j*Eq0)*np.exp(1j*omega0*delta) + \
		- RL/(x*X)*Vinf*np.exp(1j*arginf) \
)

	F2 = dId
	F3 = dIq
	F4 = np.real(expr)
	F5 = np.imag(expr)

	return [F0,F1,F2,F3,F4,F5]

def machine_noshort_quasi(z,t):
	omega = z[0]
	delta = z[1]

	expr = (\
		(( omega/x + RL/(x*X) + 1j*2/x)*(Ed0 + 1j*Eq0)*np.exp(1j*omega0*delta) - RL/(x*X)*Vinf*np.exp(1j*arginf) )/ \
	#
		( r*RL/(X*x) - 1 + 1j*( (RL + r)/x + RL/X)) )
	
	Id = np.real(expr)
	Iq = np.imag(expr)

	Pe = np.real((Ed0 + 1j*Eq0)*np.exp(1j*omega0*delta)*(Id - 1j*Iq))

	PM = PM0 - kp*omega
	F0 = (PM - Pe - D*omega)/(2*H)
	F1 = omega

	return [F0,F1]

tshort = np.linspace(0, to, 1*10**3 + 1,1)
print(tshort)
y = spi.odeint(machine_short, [0,delta0,Id0,Iq0], tshort)
omegashort = y[:,0]
deltashort = y[:,1]
Idshort = y[:,2]
Iqshort = y[:,3]
ishort = [np.real( (Idshort[k] + 1j*Iqshort[k])*np.exp(1j*omega0*tshort[k])  ) for k in range(len(tshort))]

dIshort = [ omega0/x*( (Ed0 + 1j*Eq0)*np.exp(1j*omega0*deltashort[k]) - (r + 1j*x)*(Idshort[k] + 1j*Iqshort[k])) for k in range(len(tshort)) ]

def terminal_voltage_calc_quasi(t,I,delta): return np.array([ (Ed0 + 1j*Eq0)*np.exp(1j*omega0*delta[k]) - I[k]*(r + 1j*x) for k in range(len(t))])
def terminal_power_calc_quasi(t,I,delta):
	termvol = terminal_voltage_calc_quasi(t,I,delta)
	return np.array([ termvol[k]*(np.real(I[k]) - 1j*np.imag(I[k])) for k in range(len(t))])

def terminal_voltage_calc(t,I,dI,delta): return np.array([ (Ed0 + 1j*Eq0)*np.exp(1j*omega0*delta[k]) - I[k]*(r + 1j*x) - x/omega0*dI[k] for k in range(len(t))])
def terminal_power_calc(t,I,dI,delta):
	termvol = terminal_voltage_calc(t,I,dI,delta)
	return np.array([ termvol[k]*(np.real(I[k]) - 1j*np.imag(I[k])) for k in range(len(t))])

Vshort = terminal_voltage_calc(tshort,Idshort + 1j*Iqshort,dIshort,deltashort)
Sshort = terminal_power_calc(tshort,Idshort + 1j*Iqshort,dIshort,deltashort)

y = spi.odeint(machine_short_quasi, [0,delta0], tshort)
omegashortquasi = y[:,0]
deltashortquasi = y[:,1]

expr = [((Ed0 + 1j*Eq0)*np.exp(1j*omega0*deltashortquasi[k]))/(r + 1j*x) for k in range(len(tshort))]
Idshortquasi = np.real(expr)
Iqshortquasi = np.imag(expr)

ishortquasi = [np.real( (Idshortquasi[k] + 1j*Iqshortquasi[k])*np.exp(1j*omega0*tshort[k])  ) for k in range(len(tshort))]

Vshortquasi = terminal_voltage_calc_quasi(tshort,Idshortquasi + 1j*Iqshortquasi,deltashortquasi)
Sshortquasi = terminal_power_calc_quasi(tshort,Idshortquasi + 1j*Iqshortquasi,deltashortquasi)

# Evaluating the derivaties of Id and Iq  at the end of simulation
dId0 = 10**3*(Idshort[-1] - Idshort[-2])
dIq0 = 10**3*(Iqshort[-1] - Iqshort[-2])
print(dId0,dIq0)

fig1, [ax1, ax2] = pyplot.subplots(2, 1)
fig2, [ax3, ax4] = pyplot.subplots(2, 1)
fig3, [ax5, ax6] = pyplot.subplots(2, 1)
fig4, [ax7, ax8] = pyplot.subplots(2, 1)
fig5, ax9 = pyplot.subplots(1, 1)

ax1.plot(tshort,omegashort)
ax1.plot(tshort,omegashortquasi)
ax1.set_ylabel(r'$\omega$')

ax2.plot(tshort,deltashort)
ax2.plot(tshort,deltashortquasi)
ax2.set_ylabel(r'$\delta$')

ax3.plot(tshort,Idshort)
ax3.plot(tshort,Idshortquasi)
ax3.set_ylabel(r'$I_d$')

ax4.plot(tshort,Iqshort)
ax4.plot(tshort,Iqshortquasi)
ax4.set_ylabel(r'$I_q$')

ax5.plot(tshort,np.real(Vshort))
ax5.plot(tshort,np.real(Vshortquasi))
ax5.set_ylabel(r'$V_d$')

ax6.plot(tshort,np.imag(Vshort))
ax6.plot(tshort,np.imag(Vshortquasi))
ax6.set_ylabel(r'$V_q$')

ax7.plot(tshort,np.real(Sshort))
ax7.plot(tshort,np.real(Sshortquasi))
ax7.set_ylabel(r'$P$')

ax8.plot(tshort,np.imag(Sshort))
ax8.plot(tshort,np.imag(Sshortquasi))
ax8.set_ylabel(r'$q$')

tf = 2
tnoshort = np.linspace(to, tf, 1*10**4 + 1,1)
y = spi.odeint(machine_noshort, [omegashort[-1],deltashort[-1],Idshort[-1],Iqshort[-1],dId0,dIq0], tnoshort)
omeganoshort = y[:,0]
deltanoshort = y[:,1]
Idnoshort = y[:,2]
Iqnoshort = y[:,3]
dIdnoshort = y[:,4]
dIqnoshort = y[:,5]

inoshort = [np.real( (Idnoshort[k] + 1j*Iqnoshort[k])*np.exp(1j*omega0*tnoshort[k])  ) for k in range(len(tnoshort))]

Vnoshort = terminal_voltage_calc(tnoshort,Idnoshort + 1j*Iqnoshort,dIdnoshort + 1j*dIqnoshort,deltanoshort)
Snoshort = terminal_power_calc(tnoshort,Idnoshort + 1j*Iqnoshort,dIdnoshort + 1j*dIqnoshort,deltanoshort)

y = spi.odeint(machine_noshort_quasi, [omegashortquasi[-1],deltashortquasi[-1]], tnoshort)
omeganoshortquasi = y[:,0]
deltanoshortquasi = y[:,1]

expr = [ ( ( omeganoshortquasi[k]/x + RL/(x*X) + 1j*2/x )*(Ed0 + 1j*Eq0)*np.exp(1j*omega0*deltanoshortquasi[k]) - RL/(x*X)*Vinf*np.exp(1j*arginf) ) / ( r*RL/(X*x) - 1 + 1j*( (RL + r)/x + RL/X))  for k in range(len(tnoshort))]
Idnoshortquasi = np.real(expr)
Iqnoshortquasi = np.imag(expr)

inoshortquasi = [np.real( (Idnoshortquasi[k] + 1j*Iqnoshortquasi[k])*np.exp(1j*omega0*tnoshort[k])  ) for k in range(len(tnoshort))]

Vnoshortquasi = terminal_voltage_calc_quasi(tnoshort,Idnoshortquasi + 1j*Iqnoshortquasi,deltanoshortquasi)
Snoshortquasi = terminal_power_calc_quasi(tnoshort,Idnoshortquasi + 1j*Iqnoshortquasi,deltanoshortquasi)

ax1.plot(tnoshort,omeganoshort)
ax1.plot(tnoshort,omeganoshortquasi)
ax1.set_ylabel(r'$\omega$')

ax2.plot(tnoshort,deltanoshort)
ax2.plot(tnoshort,deltanoshortquasi)
ax2.set_ylabel(r'$\delta$')

ax3.plot(tnoshort,Idnoshort)
ax3.plot(tnoshort,Idnoshortquasi)
ax3.set_ylabel(r'$I_d$')

ax4.plot(tnoshort,Iqnoshort)
ax4.plot(tnoshort,Iqnoshortquasi)
ax4.set_ylabel(r'$I_q$')

ax5.plot(tnoshort,np.real(Vnoshort))
ax5.plot(tnoshort,np.real(Vnoshortquasi))
ax5.set_ylabel(r'$V_d$')

ax6.plot(tnoshort,np.imag(Vnoshort))
ax6.plot(tnoshort,np.imag(Vnoshortquasi))
ax6.set_ylabel(r'$V_q$')

ax7.plot(tnoshort,np.real(Snoshort))
ax7.plot(tnoshort,np.real(Snoshortquasi))
ax7.set_ylabel(r'$P$')

ax8.plot(tnoshort,np.imag(Snoshort))
ax8.plot(tnoshort,np.imag(Snoshortquasi))
ax8.set_ylabel(r'$Q$')

ax9.plot(tshort,ishort,color='blue')
ax9.plot(tnoshort,inoshort,color='blue')
ax9.plot(tshort,ishortquasi,color='red')
ax9.plot(tnoshort,inoshortquasi,color='red')
ax9.set_ylabel(r'$i(t)$')

data_short = np.array([tshort,omegashort,deltashort,Idshort,Iqshort,np.real(Vshort),np.imag(Vshort),np.real(Sshort),np.imag(Sshort),ishort]).T
np.savetxt("data_omib_sim_short.csv", data_short, delimiter=",")

data_shortquasi = np.array([tshort,omegashortquasi,deltashortquasi,Idshortquasi,Iqshortquasi,np.real(Vshortquasi),np.imag(Vshortquasi),np.real(Sshortquasi),np.imag(Sshortquasi),ishortquasi]).T
np.savetxt("data_omib_sim_shortquasi.csv", data_shortquasi, delimiter=",")

data_noshort = np.array([tnoshort,omeganoshort,deltanoshort,Idnoshort,Iqnoshort,np.real(Vnoshort),np.imag(Vnoshort),np.real(Snoshort),np.imag(Snoshort),inoshort]).T
np.savetxt("data_omib_sim_noshort.csv", data_noshort, delimiter=",")

data_noshortquasi = np.array([tnoshort,omeganoshortquasi,deltanoshortquasi,Idnoshortquasi,Iqnoshortquasi,np.real(Vnoshortquasi),np.imag(Vnoshortquasi),np.real(Snoshortquasi),np.imag(Snoshortquasi),inoshortquasi]).T
np.savetxt("data_omib_sim_noshortquasi.csv", data_noshortquasi, delimiter=",")

pyplot.show()
