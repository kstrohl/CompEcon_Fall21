## S-Period Model

import numpy as np

def get_L(n_s):
    '''
    Function to compute aggregate labor supplied
    '''
    L = n_s.sum()
    return L


def get_K(b):
    '''
    Function to compute aggregate capital supplied
    '''
    K = b.sum()
    return K


def get_r(K, L, params):
    '''
    Compute the interest rate from the firm's FOC
    '''
    alpha, delta, A = params
    r = alpha * A * (L/K) ** (1 - alpha) - delta
    return r


def get_w(r, params):
    '''
    Solve for the w that is consistent with r from the firm's FOC
    '''
    alpha, delta, A = params
    w = (1 - alpha) * A * ((alpha * A) / (r + delta)) ** (alpha / (1 - alpha))
    return w


def mu_c_func(c, sigma):
    '''
    Marginal utility of consumption
    '''
    mu_c = c ** -sigma
    return mu_c


def get_c(r, w, b_s, b_sp1, n_s):
    '''
    Find consumption using the budget constraint and the choice of savings (b_sp1)
    '''
    c= np.dot(r+1,b_s) + np.dot(w, n_s) - b_sp1
    # c = ((1 + r) * b_s) + (w * n_s) - b_sp1
    return c


def mu_n_func(chi, b_param, l_tilde, n_s, nu):
    '''
    Marginal utility of labor (n)
    '''
    mu_n = chi * (b_param/l_tilde) * ((n_s/l_tilde) ** (nu-1)) * (1 - ((n_s/l_tilde) ** nu)) ** ((1-nu)/nu)
    return mu_n


# Solve for values of b and n, given r and w, from hh_foc
# Need euler equation to hold (4.25 in chapter 4), as well as MU_C = MU_N (4.24 in chapter 4)\
# Goal is to return an array of size 2*S-1 with values of b and n that satisfy above equations
def hh_foc(bn_list, r, w, params):
    '''
    Define the household first order conditions
    '''
    sigma, beta, S, chi, b_param, l_tilde, nu = params
    b_sn = bn_list
    b_sn[0] = 0
    b_s = b_sn[0:S] # from period 0 to 80
    b_sp1n = bn_list
    b_sp1n[S] = 0
    b_sp1 = b_sp1n[1:S+1] # from period 1 to 80
    b_sp1n[-1] = 0
    n_s = bn_list[S+1:2*S+1]
    c = get_c(r, w, b_s, b_sp1, n_s)
    mu_c = mu_c_func(c, sigma)
    mu_n = mu_n_func(chi, b_param, l_tilde, n_s, nu)
    euler_error = mu_c[:-1] - beta * (1+r) * mu_c[1:]
    euler_error_2 = w * mu_c - mu_n
    return euler_error, euler_error_2
