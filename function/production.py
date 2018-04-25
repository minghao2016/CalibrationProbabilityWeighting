"""
Purpose: Functions for calibration
Author: Yuhao Zhu
"""

import math
import numpy as np

"""
Production function:
"""

def f_P_T(u, P_0, r_f, d, s, T):
    """
    Calculate the stock price at the time T.
    s is period standard deviation.
    sigma is the overal standard deviation.
    """
    sigma = s * math.sqrt(T)
    sigma_2 = math.pow(sigma, 2)
    mu = math.log(P_0) + (r_f - d) * T - sigma_2 / 2
    P_T = math.exp(mu + u * sigma)
    return P_T

def f_ln_P_T(u, P_0, r_f, d, s, T):
    """
    Calcluate the logrithmic stock price at the time T.
    """
    sigma = s * math.sqrt(T)
    sigma_2 = math.pow(sigma, 2)
    mu = math.log(P_0) + (r_f - d) * T - sigma_2 / 2
    ln_P_T = mu + u * sigma
    return ln_P_T

def f_P_T_to_P_0(u, r_f, d, s, T):
    """
    Calcluate the ratio between stock price at time T and the time 0.
    """
    sigma = s * math.sqrt(T)
    sigma_2 = math.pow(sigma, 2)
    P_T_to_P_0 = math.exp((r_f - d) * T - sigma_2 / 2 + u * sigma)
    return P_T_to_P_0
