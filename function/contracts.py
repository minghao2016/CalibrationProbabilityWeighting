
"""
Purpose: Functions for calibration
Author: Yuhao Zhu
"""

import math
import numpy as np

"""
Piece-wise linear contract.
Function are with suffix _pc.
"""

def f_W_T_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, wealth_min=0.01):
    """
    The wealth to the agent given the stock price at time T.
    wealth_min: the minimum wealth of the CEO. Default value 0.01.
    """
    P_T = f_P_T(v, P_0, r_f, d, s, T)
    W_T = (phi + wealth) * math.exp(r_f * T) + n_s * math.exp(d * T) * P_T + n_o * max(P_T - K, 0)
    W_T = max(W_T, wealth_min)
    return W_T

def f_W_0_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, wealth_min=0.01):
    """
    The wealth to the agent given the stock price at time 0.
    wealth_min: the minimum wealth of the CEO. Default value 0.01.
    """
    W_T = f_W_T_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, wealth_min=wealth_min)
    W_0 = W_T * math.exp(-r_f * T)
    return W_0