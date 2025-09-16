from sympy import *
import sympy
init_printing()

from sympy.abc import t
x = Function('x')(t)
w = Function('w')(t)

r, L, Lp, RL = symbols('r L Lp RL')

def ndpo(n,freq,signal):
	# OBS: n is supposed equal or greater than zero
	if n > 0: return Derivative(ndpo(n-1,freq,signal),t) + 1j*freq*ndpo(n-1,freq,signal)
	else: return signal

k = 2
print(collect(simplify(ndpo(k,w,x)),[Derivative(x,t,i) for i in range(k+1)]))

expr = collect(expand(simplify(L*Lp/RL*ndpo(2,w,x) + (r*L/RL + L + Lp)*ndpo(1,w,x) + r*x)),x)

print(sympy.latex(
	expr,
	symbol_names = {
	L: "L",
	Lp: "L'",
	r: "r",
	RL: "R_L"
	},
	imaginary_unit='j',
))
