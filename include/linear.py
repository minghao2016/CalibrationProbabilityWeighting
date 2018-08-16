"""
Purpose: Functions for calibration
Author: Yuhao Zhu
"""

import math
import numpy as np
from include.basic import *
from scipy.stats import norm
from scipy import integrate


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


"""
Derivatives.
Piece-wise linear contract.
Function are with suffix _pc.
"""

def f_W_T_to_P_T_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K):
    """
    The derivative of W_T with respect to P_T.
    """
    P_T = f_P_T(v, P_0, r_f, d, s, T)
    if P_T > K:
        value = n_s * math.exp(d * T) + n_o
    else:
        value = n_s * math.exp(d * T)
    return value

def f_W_0_to_P_T_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K):
    """
    The derivative of W_0 with respect to P_T.
    """
    value = f_W_T_to_P_T_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K)
    value = value * math.exp(-r_f * T)
    return value

"""
Cost functions
Piece-wise linear contract.
Function are with suffix _pc.
"""

def f_cost_T_pc(u, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K):
    P_T = f_P_T(u, P_0, r_f, d, s, T)
    cost = phi * math.exp(r_f * T) + n_s * math.exp(d * T) * P_T + n_o * max(P_T - K, 0)
    return cost
    
def f_cost_T_integral_pc(u, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K):
    """
    Integrate of the wage discounted at time 0: cost function.
    """
    func = f_cost_T_pc(u, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K) * norm.pdf(u)
    return func

def f_E_cost_T_pc(variables, parameters):
    phi, n_s, n_o = variables
    P_0, r_f, d, s, T, wealth, K = parameters
    E_cost_T = integrate.quad(f_cost_T_integral_pc, -20, 20,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K))[0]
    return E_cost_T

def f_E_cost_0_pc(variables, parameters):
    """
    phi, n_s, n_o = variables
    P_0, r_f, d, s, T, wealth, K = parameters
    """
    r_f = parameters[1]
    T = parameters[4]
    E_cost_T = f_E_cost_T_pc(variables, parameters)
    E_cost_0 = E_cost_T * math.exp(-r_f * T)
    return E_cost_0

"""
Expected utility.
Piece-wise linear contract.
Function are with suffix _pc.
use v with tranformation.
"""

def f_U_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, gamma, wealth_min=0.01):
    W_T = f_W_T_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, wealth_min=wealth_min)
    value = f_U_function(W_T, gamma)
    return value

def f_U_integral_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, gamma, eta_s, eta_m, wealth_min=0.01):
    func = f_U_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, gamma, wealth_min=0.01) * norm.pdf(v, eta_m, eta_s)
    return func

def f_E_U_pc(variables_pc, parameters_pc, gamma, eta_s, eta_m, wealth_min=0.01):
    phi, n_s, n_o = variables_pc
    P_0, r_f, d, s, T, wealth, K = parameters_pc
    value = integrate.quad(f_U_integral_pc, -20, 20,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, gamma, eta_s, eta_m, wealth_min))[0]
    return value


"""
Expected utility-adjsted PPS.
Piece-wise linear contract.
Function are with suffix _pc.
use v with tranformation.
"""

def f_UPPS_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, gamma, wealth_min=0.01):
    """
    Utility-adjusted PPS (UPPS)
    The incentive constraint of the agent (UPPS).
    """
    W_T = f_W_T_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, wealth_min=wealth_min)
    value = pow(W_T, -gamma) * f_W_T_to_P_T_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K) * f_P_T_to_P_0(v, r_f, d, s, T)
    return value

def f_UPPS_integral_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, gamma, eta_s, eta_m, wealth_min=0.01):
    """
    UPPS integral.
    """
    func = f_UPPS_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, gamma, wealth_min=wealth_min) * norm.pdf(v, eta_m, eta_s)
    return func

def f_E_UPPS_pc(variables_pc, parameters_pc, gamma, eta_s, eta_m, wealth_min=0.01):
    """
    Expected utility-adjusted PPS.
    """
    phi, n_s, n_o = variables_pc
    P_0, r_f, d, s, T, wealth, K = parameters_pc
    value = integrate.quad(f_UPPS_integral_pc, -20, 20,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, gamma, eta_s, eta_m, wealth_min))[0]
    return value


"""
Distance metrics for pc
"""

def f_distance(u, P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt, wealth_min=0.01):
    W_0 = f_W_0_pc(u, P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, wealth_min=wealth_min)
    W_0_opt = f_W_0_pc(u, P_0, r_f, d, sigma, T, phi_opt, n_s_opt, n_o_opt, K, wealth, wealth_min=wealth_min)
    value = abs(W_0 - W_0_opt) / W_0
    return value

def f_distance_integral(u, P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt, wealth_min=0.01):
    func = f_distance(u, P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt, wealth_min=wealth_min) * norm.pdf(u)
    return func

def f_E_distance(P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt, wealth_min=0.01):
    value = integrate.quad(f_distance_integral, -20, 20,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                          args=(P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt, wealth_min))[0]
    return value