"""
Purpose: Functions for calibration
Author: Yuhao Zhu
"""

import math
import numpy as np
from include.basic import *
from include.basic import f_W_0_pc
from scipy.stats import norm
from scipy import integrate

"""
Define sigma-mu transformation.
"""

def f_sigma_A(sigma, mu, eta_s, eta_m):
    sigma_A = eta_s * sigma
    return sigma_A

def f_mu_A(sigma, mu, eta_s, eta_m):
    mu_A = mu + eta_m * sigma
    return mu_A

def f_beta(P_0, r_f, d, s, T, eta_s, eta_m):
    """
    Calculate the beta's.
    """
    sigma = s * math.sqrt(T)
    sigma_2 = math.pow(sigma, 2)
    mu = math.log(P_0) + (r_f - d) * T - sigma_2 / 2
    sigma_A = f_sigma_A(sigma, mu, eta_s, eta_m)
    sigma_A_2 = math.pow(sigma_A, 2)
    mu_A = f_mu_A(sigma, mu, eta_s, eta_m)
    beta_0 = math.pow(mu,2)/(2*sigma_2) - math.pow(mu_A,2)/(2*sigma_A_2)
    beta_1 = -(mu/(sigma_2) - mu_A/(sigma_A_2))
    beta_2 = 1/(2*sigma_2) - 1/(2*sigma_A_2)
    beta = (beta_0, beta_1, beta_2)
    return beta


"""
General contract.
Function are with suffix _gc.
"""

def f_W_T_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min=0.01):
    """
    The wage payment (optimal) to the agent given the stock price discounted to present.
    De factor parameters are eta_s, eta_m, alpha_0, alpha_1.
    """
    (beta_0, beta_1, beta_2) = f_beta(P_0, r_f, d, s, T, eta_s, eta_m)
    ln_P_T = f_ln_P_T(u, P_0, r_f, d, s, T)

    R_T = math.exp(beta_2 * math.pow(ln_P_T, 2) + beta_1 * ln_P_T + beta_0) * (alpha_1 * ln_P_T + alpha_0)

    if R_T > math.pow(wealth_min, gamma):
        W_T = math.pow(R_T, 1/gamma)
    else:
        W_T = wealth_min
    return W_T

def f_W_0_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min=0.01):
    """
    The wealth to the agent given the stock price at time 0.
    wealth_min: the minimum wealth of the CEO. Default value 0.01.
    """
    W_T = f_W_T_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min)
    W_0 = W_T * math.exp(-r_f * T)
    return W_0


"""
Derivatives.
Piece-wise linear contract.
Function are with suffix _gc.
"""

def f_W_T_to_P_T_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min=0.01):
    """
    The derivative of W_T_gc with respect to P_T.
    """
    beta_0, beta_1, beta_2 = f_beta(P_0, r_f, d, s, T, eta_s, eta_m)
    
    P_T = f_P_T(v, P_0, r_f, d, s, T)
    ln_P_T = f_ln_P_T(v, P_0, r_f, d, s, T)
    
    if ln_P_T > - beta_1 / (2 * beta_2):
        W_T = f_W_T_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min)
        R_T = math.exp(beta_2 * math.pow(ln_P_T, 2) + beta_1 * ln_P_T + beta_0) * (alpha_1 * ln_P_T + alpha_0)
        if R_T > math.pow(wealth_min, gamma):
            value = 1/gamma * math.pow(W_T, 1-gamma) * (math.pow(W_T, gamma) * (beta_2 * 2 * ln_P_T / P_T + beta_1 / P_T) 
                    + math.exp(beta_2 * math.pow(ln_P_T, 2) + beta_1 * ln_P_T + beta_0) * alpha_1 / P_T)  
        else:
            value = 0
    else:
        value = 0
    return value

def f_W_0_to_P_T_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min=0.01):
    """
    The derivative of W_0 with respect to P_T.
    """
    value = f_W_T_to_P_T_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min)
    value = value * math.exp(-r_f * T)
    return value


"""
Cost functions
Piece-wise linear contract.
Function are with suffix _pc.
"""

def f_cost_T_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min=0.01):
    """
    Gamma is needed.
    """
    W_T = f_W_T_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min)
    cost = W_T - wealth * math.exp(r_f * T)
    return cost

    
def f_cost_T_integral_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min=0.01):
    """
    Integrate of the wage discounted at time T: cost function.
    """
    func = f_cost_T_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min) * norm.pdf(u)
    return func


def f_E_cost_T_gc(variables, parameters):
    alpha_0, alpha_1 = variables
    P_0, r_f, d, s, T, wealth, gamma, eta_s, eta_m, wealth_min = parameters
    E_cost_T = integrate.quad(f_cost_T_integral_gc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min))[0]
    return E_cost_T

def f_E_cost_0_gc(variables, parameters):
    """
    alpha_0, alpha_1 = variables
    P_0, r_f, d, s, T, wealth, gamma, eta_s, eta_m, wealth_min = parameters
    """
    r_f = parameters[1]
    T = parameters[4]
    E_cost_T = f_E_cost_T_gc(variables, parameters)
    E_cost_0 = E_cost_T * math.exp(-r_f * T)
    return E_cost_0


"""
Expected utility.
Piece-wise linear contract.
Function are with suffix _pc.
use v with tranformation.
"""

def f_U_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min=0.01):
    W_T = f_W_T_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min)
    value = f_U_function(W_T, gamma)
    return value

def f_U_integral_gc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, gamma, eta_s, eta_m, wealth_min=0.01):
    func = f_U_gc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K, gamma, wealth_min) * norm.pdf(v, eta_m, eta_s)
    return func

def f_E_U_gc(variables_gc, parameters_gc, gamma, eta_s, eta_m, wealth_min=0.01):
    alpha_0, alpha_1 = variables_gc
    P_0, r_f, d, s, T, wealth = parameters_gc
    value = integrate.quad(f_U_integral_gc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min))[0]
    return value


"""
Expected utility-adjsted PPS.
Piece-wise linear contract.
Function are with suffix _pc.
use v with tranformation.
"""

def f_UPPS_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min=0.01):
    """
    Utility-adjusted PPS (UPPS)
    The incentive constraint of the agent (UPPS).
    """
    W_T = f_W_T_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min)
    W_T_to_P_T = f_W_T_to_P_T_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min)
    P_T_to_P_0 = f_P_T_to_P_0(v, r_f, d, s, T)
    
    value = math.pow(W_T, -gamma) * W_T_to_P_T * P_T_to_P_0 
    return value


def f_UPPS_integral_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min=0.01):
    """
    UPPS integral.
    """
    func = f_UPPS_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min) * norm.pdf(v, eta_m, eta_s)
    return func

"""
==================================================================================
Pause here.
==================================================================================
"""

def f_E_UPPS_gc(variables_gc, parameters_gc, gamma, eta_s, eta_m, wealth_min=0.01):
    """
    Expected utility-adjusted PPS.
    """
    alpha_0, alpha_1 = variables_gc
    P_0, r_f, d, s, T, wealth = parameters_gc
    value = integrate.quad(f_UPPS_integral_gc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, alpha_0, alpha_1, gamma, eta_s, eta_m, wealth_min))[0]
    return value


"""
Distance metrics for general contract
"""

def f_distance_gc(u, P_0, r_f, d, s, T, phi, n_s, n_o, K, wealth, alpha_0_opt, alpha_1_opt, gamma, eta_s, eta_m, wealth_min=0.01):
    W_0 = f_W_0_pc(u, P_0, r_f, d, s, T, phi, n_s, n_o, K, wealth, wealth_min=wealth_min)
    W_0_opt = f_W_0_gc(u, P_0, r_f, d, s, T, wealth, alpha_0_opt, alpha_1_opt, gamma, eta_s, eta_m, wealth_min=0.01)
    value = abs(W_0 - W_0_opt) / W_0
    return value

def f_distance_integral_gc(u, P_0, r_f, d, s, T, phi, n_s, n_o, K, wealth, alpha_0_opt, alpha_1_opt, gamma, eta_s, eta_m, wealth_min=0.01):
    func = f_distance_gc(u, P_0, r_f, d, s, T, phi, n_s, n_o, K, wealth, alpha_0_opt, alpha_1_opt, gamma, eta_s, eta_m, wealth_min) * norm.pdf(u)
    return func

def f_E_distance_gc(P_0, r_f, d, s, T, phi, n_s, n_o, K, wealth, alpha_0_opt, alpha_1_opt, gamma, eta_s, eta_m, wealth_min=0.01):
    value = integrate.quad(f_distance_integral_gc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                          args=(P_0, r_f, d, s, T, phi, n_s, n_o, K, wealth, alpha_0_opt, alpha_1_opt, gamma, eta_s, eta_m, wealth_min))[0]
    return value