from sympy import *
import sympy
init_printing()
sigma, RB1, RB2, rpi, ro, gm, CL, CB, RL, RE, RC = symbols('sigma RB1 RB2 rpi ro gm CL CB RL RE RC')

CBv = 10**(-3)
CLv = 10**(-3)
REv = 10**1
RCv = 10**3
RB1v = 10**3
RB2v = 10**4
rpiv = 10**6
rov = 10**6
gmv = 10**(-6)
RLv = 10**6

M = Matrix([
	[-sigma*CB - 1/RB1 - 1/RB2 - 1/rpi, 1/rpi, 0, 0],
	[1/rpi + gm, -1/rpi - gm - 1/RE - 1/ro, 1/ro, 0],
	[-gm, gm + 1/ro, -1/ro -1/RC - sigma*CL, sigma*CL],
	[0,0,sigma*CL, -sigma*CL - 1/RL]
])

Mi = M.inv()
print(sympy.latex(
	simplify(Mi[0,0]),
	symbol_names = {
	RB1: "R_{B1}",
	RB2: "R_{B2}",
	rpi: "r_{\\pi}",
	ro: "r_o",
	CL: "C_L",
	CB: "C_B",
	gm: "g_m",
	RE: "R_E",
	RL: "R_L",
	RC: "R_{C}",
	}
))

Msubst = M.subs([(RB1,RB1v), (RB2,RB2v), (rpi,rpiv), (ro,rov), (gm,gmv), (CL,CLv), (CB,CBv), (RL,RLv), (RE,REv), (RC,RCv)])
print(sympy.latex(
	Msubst.inv(),
	))

