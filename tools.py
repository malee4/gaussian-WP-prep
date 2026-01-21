import numpy as np
from sympy import sin, cos, Symbol, Array, Eq, lambdify, Float
from sympy.solvers import solve, nsolve

def get_unitary_thetas(d, c=None, tol=None):
	""" 
	Calculates the CRy rotations for a log-scaling unitary circuit for state preparation on 
	a device with all-to-all connectivity.

	args:
		d (int):		wavepacket size
		c (np.ndarray):	expected coefficients. If None is passed, the uniform probability distribution is assumed.
	
	returns:
		thetas (np.ndarray): array of CRy rotations, where each index corresponds to its order in the circuit.		
	"""
	# sympy
	if c is None:
		c = np.full(d, 1 / np.sqrt(d))
	
	if tol is None:
		tol = 1e-6

	ry_theta = [Symbol(f't{i}') for i in range(0, d-1)]
	# signs = [Symbol(f't{i}') for i in range(1, d)]

	# construct \Theta
	big_theta = np.ones((d, d), dtype=object)

	for n in range(d):
		for j in range(0, min(2**n, d - 2**n)):
			big_theta[j][2**n + j] = cos(ry_theta[2**n + j - 1] / 2)
			big_theta[2**n + j][2**n + j] = sin(ry_theta[2**n + j - 1] / 2)

	for n_p in range(1, d):
		for n in range(n_p, d):
			for j in range(0, min(2**n, d - 2**n) - 1 + 1):
				for jp in range(j, -1, -2):
					big_theta[2**n + j][jp] = big_theta[2**(n - n_p) + j - 1][2**(n - n_p) + jp - 1]
					big_theta[2**n + j][2**(n - n_p) + jp] = big_theta[j][2**(n - n_p) + jp]

	equation_sys = []

	# # sort the coefficients in descending order
	# c_sorted = np.sort(c)[::-1]
	# indices = np.argsort(c)[::-1]

	# obtain list of equations
	for i in range(len(big_theta)):
		# prod
		row = big_theta[i]
		equation = 1
		for exp in row:
			equation *= exp

		equation = Eq(equation, Float(c[i]))
		# add to list of equations
		equation_sys.append(equation)

	guess = np.ones(d-1) # something arbitrary
	sol = nsolve(equation_sys, ry_theta, guess, tol=tol)
	sol = np.array(sol, dtype=float).T[0] # reshape

	# # undo the sorting
	# reordered_indices = np.argsort(indices)
	# sol = sol[reordered_indices]

	return sol