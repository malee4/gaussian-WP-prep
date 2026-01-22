import numpy as np
from sympy import sin, cos, Symbol, Array, Eq, lambdify, Float
from sympy.solvers import solve, nsolve

from pytket.circuit import Circuit
from pytket.circuit.logic_exp import (
	if_bit
)

def get_2d_gaussian_coeff(lx, ly, k0, x0, sigma, epsilon=0.0):
	"""
	Docstring for get_2d_gaussian_coeff
	
	:param lx: Description
	:param ly: Description
	:param k0: Description
	:param x0: Description
	:param sigma: Description
	:param epsilon: Description
	"""
	# Define momentum eigenstates
	if lx % 2 == 0:
		kxs = np.arange(- int(lx / 2) + 1, int(lx / 2) + 1) * 2 * np.pi / lx
	else:
		kxs = np.arange(- int((lx + 1) / 2) + 1, int(lx / 2) + 1) * 2 * np.pi / lx

	if ly % 2 == 0:
		kys = np.arange(- int(ly / 2) + 1, int(ly / 2) + 1) * 2 * np.pi / ly
	else:
		kys = np.arange(- int((ly + 1) / 2) + 1, int(ly / 2) + 1) * 2 * np.pi / ly
		
	# Calculate the coefficients for any k-pairing
	x_coeff = np.zeros((lx, ly), dtype=np.complex128)

	# Find the values in the exponent of the coefficient
	x_coeff += (-1j * ((kxs * x0[0]).reshape(-1, 1) + (kys * x0[1])) # all possible combinations
				+ (-1 * (kxs - k0[0]) ** 2 / (4 * sigma[0] ** 2)).reshape(-1, 1) # add kx norm
				+ (-1 * (kys - k0[1]) ** 2 / (4 * sigma[1] ** 2)) # add ky norm
			) 
	x_coeff = np.exp(x_coeff)

	# Calculate the position-specific coefficients
	c = np.zeros(lx * ly, dtype=np.complex128)
	for nx in range(lx):
		for ny in range(ly):
			index = ny * lx + nx
			
			# Select all possible pairings of kx ky
			ks = np.sum(np.stack(np.meshgrid(kys, kxs), -1) * np.array([ny, nx]), axis=2)
			
			ks_sine = np.sin(ks)
			ks_cosine = np.cos(ks)

			ks_complex = ks_cosine + 1j * ks_sine
			
			# multiply by x coeff and sum
			c[index] = np.sum(x_coeff * ks_complex)

	# Normalize
	c = c / np.linalg.norm(c)

	# Drop small values less than cutoff
	mask = np.abs(c) ** 2 > epsilon 	# bit mask for all |c_n| < epsilon
	c = c * mask 						# apply bit mask
	c = c/np.sqrt(np.vdot(c, c))		# renormalize
	return c
	

def get_unitary_thetas(N, c=None, tol=None):
	""" 
	Calculates the CRy rotations for a log-scaling unitary circuit for state preparation on 
	a device with all-to-all connectivity.
	For complex coefficients, an additional step using $R_Z$ gates in the circuit must be performed.

	args:
		N (int):		wavepacket size
		c (np.ndarray):	expected real coefficients. If None is passed, the uniform probability distribution is assumed.
	
	returns:
		thetas (np.ndarray): array of CRy rotations, where each index corresponds to its order in the circuit.		
	"""
	# sympy
	if c is None:
		c = np.full(N, 1 / np.sqrt(N))
	
	if tol is None:
		tol = 1e-6

	ry_theta = [Symbol(f't{i}') for i in range(0, N-1)]

	# construct \Theta
	big_theta = np.ones((N, N), dtype=object)

	for n in range(N):
		for j in range(0, min(2**n, N - 2**n)):
			big_theta[j][2**n + j] = cos(ry_theta[2**n + j - 1] / 2)
			big_theta[2**n + j][2**n + j] = sin(ry_theta[2**n + j - 1] / 2)

	for n_p in range(1, N):
		for n in range(n_p, N):
			for j in range(0, min(2**n, N - 2**n) - 1 + 1):
				for jp in range(j, -1, -2):
					big_theta[2**n + j][jp] = big_theta[2**(n - n_p) + j - 1][2**(n - n_p) + jp - 1]
					big_theta[2**n + j][2**(n - n_p) + jp] = big_theta[j][2**(n - n_p) + jp]

	equation_sys = []

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

	guess = np.ones(N-1) # something arbitrary
	sol = nsolve(equation_sys, ry_theta, guess, tol=tol)
	sol = np.array(sol, dtype=float).T[0] # reshape

	return sol

def fused_log_subcircuit_pytket(N, c=None):
	""" 
	Returns the unitary subcircuit for the MCM-FF-1 method.

	args:
		N (int):		wavepacket size
		c (np.ndarray):	expected coefficients. If None is passed, the uniform probability distribution is assumed.
	
	returns:
		qc (Circuit): unitary circuit that prepares the specified coefficients.		
	"""
	thetas = get_unitary_thetas(N, c)

	qc = Circuit(name='log subcircuit')
	qr = qc.add_q_register('q', N)

	qc.X(qr[0])

	# log scaling: the following code is from Roland
	for i in range(int(np.ceil(np.log2(N)))):
		# circuits that can run in parallel
		for j in range(2**i):
			if j + 2**i < N:
				qc.CRy(thetas[j + 2**i - 1] / np.pi, qr[j], qr[j + 2**i])
				qc.CX(qr[j + 2**i], qr[j])
	return qc