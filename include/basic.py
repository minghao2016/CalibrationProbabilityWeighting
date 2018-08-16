"""
Purpose: Basic functions for calibration
Author: Yuhao Zhu
"""

import math
import numpy as np
from scipy.stats import norm
from scipy import integrate

"""
Production function:
"""

def f_P_T(u, P_0, r_f, d, s, T):
    """
    Calculate the stock price at the time T.
    s is period standard deviation.
    sigma is the overall standard deviation.
    """
    sigma = s * math.sqrt(T)
    sigma_2 = math.pow(sigma, 2)
    mu = math.log(P_0) + (r_f - d) * T - sigma_2 / 2
    P_T = math.exp(mu + u * sigma)
    return P_T

def f_ln_P_T(u, P_0, r_f, d, s, T):
    """
    Calculate the logarithmic stock price at the time T.
    """
    sigma = s * math.sqrt(T)
    sigma_2 = math.pow(sigma, 2)
    mu = math.log(P_0) + (r_f - d) * T - sigma_2 / 2
    ln_P_T = mu + u * sigma
    return ln_P_T

def f_P_T_to_P_0(u, r_f, d, s, T):
    """
    Calculate the ratio between stock price at time T and the time 0.
    """
    sigma = s * math.sqrt(T)
    sigma_2 = math.pow(sigma, 2)
    P_T_to_P_0 = math.exp((r_f - d) * T - sigma_2 / 2 + u * sigma)
    return P_T_to_P_0


"""
Utility function.
"""
def f_U_function(W, gamma):
    """
    The CRRA utility function.
    Default gamma is 3.
    """
    if W > 0:
        if gamma == 1:
            utility = math.log(W)
        elif gamma >= 0:
            # Minus 1 or not.
            utility = math.pow(W, 1 - gamma) / (1 - gamma)
        else:
            print('The risk aversion parameter should be non-negative numbers.')
    else:
        print('The wealth should be non-negative. Now {}.'.format(W))
        utility = 0
    return utility