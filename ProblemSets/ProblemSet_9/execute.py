import SS as ss
import numpy as np


# Set parameters
S = 80
beta = 0.8
sigma = 1.5
l_tilde = 1.0
b_param = .501
nu = 1.554
chi = np.ones(S)
A = 1.0
alpha = 0.3
delta = 0.1

params = alpha, delta, A, sigma, beta, S, chi, b_param, l_tilde, nu

# Make initial guesses
r_guess = 0.1
bn_guesses = np.ones((2 * S + 1), dtype=int)


r_ss, L_ss, K_ss, success, euler_errors = ss.ss_solver(
    r_guess, bn_guesses, params)

print('The SS interest rate is ', r_ss, 'Did we find the solution?', success)