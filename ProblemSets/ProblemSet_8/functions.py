# Import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import interpolate

# Create functions
def utility(c, x, alpha, sigma):
    '''
    Per-period utility function
    '''
    U = (c ** (1 - sigma)) / (1 - sigma) + alpha * (x ** (1 - sigma)) / (1 - sigma)
    
    return U


def mu_c(c, sigma): 
    '''
    Marginal utility function for "green" investment
    '''
    MU_C = c ** -sigma
    
    return MU_C


def mu_c_inv(c, sigma): 
    MU_C = c ** (- 1 / sigma)

    return MU_C


def mu_x(x, alpha, sigma): 
    '''
    Marginal utility function for business investment
    '''
    MU_X = alpha * x ** -sigma

    return MU_X


def new_x(c, alpha, sigma): 
    '''
    Equation determining x given choice of c
    '''
    new_x = alpha ** (1 / sigma) * c

    return new_x


def coleman_egm(phi, z_prime_grid, params):
    '''
    The Coleman operator, which takes an existing guess phi of the
    optimal consumption policy and computes and returns the updated function
    Kphi on the grid points.
    '''
    alpha, sigma, beta = params

    # Initialize C and X
    C = np.empty_like(z_prime_grid)
    X = np.empty_like(z_prime_grid)
    
    # Solve for updated values
    for i, z_prime in enumerate(z_prime_grid):
        C[i] = mu_c_inv(beta * mu_c(phi(z_prime), sigma), sigma)
        X[i] = new_x(C[i], alpha, sigma)

    # Determine endogenous grid
    z_grid = z_prime_grid + C + X

    # Update policy function
    Kphi = interpolate.interp1d(z_grid, C, kind='linear', fill_value='extrapolate')
    return Kphi