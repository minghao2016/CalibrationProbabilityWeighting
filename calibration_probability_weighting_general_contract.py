# ### Optimal contract shape

# Set eta_s eta_m and gamma as global variables.

# In[ ]:

def f_sigma_A(sigma, mu):
    sigma_A = eta_s * sigma
    return sigma_A

def f_mu_A(sigma, mu):
    mu_A = mu + eta_m * sigma
    return mu_A


# In[ ]:

def f_beta(P_0, r_f, d, s, T):
    sigma = s * math.sqrt(T)
    sigma_2 = math.pow(sigma, 2)
    mu = math.log(P_0) + (r_f - d) * T - sigma_2 / 2
    sigma_A = f_sigma_A(sigma, mu)
    sigma_A_2 = math.pow(sigma_A, 2)
    mu_A = f_mu_A(sigma, mu)
    beta_0 = math.pow(mu,2)/(2*sigma_2) - math.pow(mu_A,2)/(2*sigma_A_2)
    beta_1 = -(mu/(sigma_2) - mu_A/(sigma_A_2))
    beta_2 = 1/(2*sigma_2) - 1/(2*sigma_A_2)
    beta = (beta_0, beta_1, beta_2)
    return beta


# \begin{align*}
# W_{T}=\left[\exp\left(\beta_{2}\left(\ln P_{T}\right)^{2}+\beta_{1}\ln P_{T}+\beta_{0}\right)\left(\alpha_{1}\ln P_{T}+\alpha_{0}\right)\right]^{1/\gamma}
# \end{align*}

# wealth_min is the minimum wealth of the CEO. If there is punishment, the CEO's wealth can be set to close to zero, so wealth_min = 0.000001. If there is no punishment, the CEO's wealth can be set to the initial wealth times risk-free rate.

# In[ ]:

def f_W_T_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1):
    """
    The wage payment (optimal) to the agent given the stock price discounted to present.
    De factor parameters are eta_s, eta_m, alpha_0, alpha_1.
    """
        
    (beta_0, beta_1, beta_2) = f_beta(P_0, r_f, d, s, T)
    
    P_T = f_P_T(u, P_0, r_f, d, s, T)
    ln_P_T = f_ln_P_T(u, P_0, r_f, d, s, T)
    
    # If ln_P_T is smaller than certain shreshold, force it to be the shreshold.
    ln_P_T = max(ln_P_T, - beta_1 / (2 * beta_2))
#     print('u: ', end='')
#     print(u)
#     print(beta_2 * math.pow(ln_P_T, 2) + beta_1 * ln_P_T + beta_0)

    R_T = math.exp(beta_2 * math.pow(ln_P_T, 2) + beta_1 * ln_P_T + beta_0) * (alpha_1 * ln_P_T + alpha_0)
    
    if R_T > math.pow(wealth_min, gamma):
        W_T = math.pow(R_T, 1/gamma)
    else:
        W_T = wealth_min
    
    return W_T


# In[ ]:

def f_W_T_to_P_T_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1):
    """
    The derivative of W_T_gc with respect to P_T.
    """
    (beta_0, beta_1, beta_2) = f_beta(P_0, r_f, d, s, T)
    
    P_T = f_P_T(u, P_0, r_f, d, s, T)
    ln_P_T = f_ln_P_T(u, P_0, r_f, d, s, T)
    
    if ln_P_T > - beta_1 / (2 * beta_2):
        W_T = f_W_T_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1)
        R_T = math.exp(beta_2 * math.pow(ln_P_T, 2) + beta_1 * ln_P_T + beta_0) * (alpha_1 * ln_P_T + alpha_0)
        if R_T > math.pow(wealth_min, gamma):
            value = 1/gamma * math.pow(W_T, 1-gamma) * (math.pow(W_T, gamma) * (beta_2 * 2 * ln_P_T / P_T + beta_1 / P_T) 
                    + math.exp(beta_2 * math.pow(ln_P_T, 2) + beta_1 * ln_P_T + beta_0) * alpha_1 / P_T)  
        else:
            value = 0
    else:
        value = 0
    return value


# ## 2.2. Principal related functions

# $u$ based random variable.  
# Transfer all variables in the expression of $u$.

# ### cost for piecewise linear contract

# Note that T is advanced! P_0, r_f, d, s, T, K, wealth, gamma = parameters  
# use standard normal u as the pdf in integral.

# In[ ]:

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
    value = integrate.quad(f_cost_T_integral_pc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K))[0]
    return value

def f_E_cost_0_pc(variables, parameters):
    phi, n_s, n_o = variables
    P_0, r_f, d, s, T, wealth, K = parameters
    value = integrate.quad(f_cost_T_integral_pc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K))[0]
    value = value * math.exp(-r_f * T)
    return value


# ### cost for optimal contract

# In[ ]:

def f_cost_T_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1):
    P_T = f_P_T(u, P_0, r_f, d, s, T)
    W_T = f_W_T_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1)
    cost = W_T - wealth * math.exp(r_f * T)
    return cost
    
def f_cost_T_integral_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1):
    """
    Integrate of the wage discounted at time 0: cost function.
    """
    func = f_cost_T_gc(u, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1) * norm.pdf(u)
    return func

def f_E_cost_T_gc(variables_gc, parameters_gc):
    alpha_0, alpha_1 = variables_gc
    P_0, r_f, d, s, T, wealth, K = parameters_gc
    value = integrate.quad(f_cost_T_integral_gc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, alpha_0, alpha_1))[0]
    return value

def f_E_cost_0_gc(variables_gc, parameters_gc):
    alpha_0, alpha_1 = variables_gc
    P_0, r_f, d, s, T, wealth, K = parameters_gc
    value = integrate.quad(f_cost_T_integral_gc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, alpha_0, alpha_1))[0]
    value = value * math.exp(-r_f * T)
    return value


# ## 2.3. Agent related functions

# \begin{align*}
# V(W_T) = \mathbb{E}\left[\frac{W_T^{1-\gamma}}{1-\gamma}\right]
# \end{align*}

# In[ ]:

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


# ### Expected utility for piecewise
# use v with tranformation.

# In[ ]:

def f_U_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K):
    W_T = f_W_T_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K)
    value = f_U_function(W_T, gamma)
    return value

def f_U_integral_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K):
    func = f_U_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K) * norm.pdf(v, eta_m, eta_s)
    return func

def f_E_U_pc(variables_pc, parameters_pc):
    phi, n_s, n_o = variables_pc
    P_0, r_f, d, s, T, wealth, K = parameters_pc
    value = integrate.quad(f_U_integral_pc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K))[0]
    return value


# ### Expected utility for general contract

# In[ ]:

def f_U_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1):
    W_T = f_W_T_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1)
    value = f_U_function(W_T, gamma)
    return value

def f_U_integral_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1):
    func = f_U_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1) * norm.pdf(v, eta_m, eta_s)
    return func

def f_E_U_gc(variables_gc, parameters_gc):
    alpha_0, alpha_1 = variables_gc
    P_0, r_f, d, s, T, wealth, K = parameters_gc
    value = integrate.quad(f_U_integral_gc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, alpha_0, alpha_1))[0]
    return value


# ### Expected utility-adjusted PPS

# \begin{align*}
# EUPPS=EUD= & \mathbb{\mathbb{E}}\left[\frac{dV(W_{0})}{dW_{0}}\cdot\frac{dW_{0}}{dP_{T}}\cdot\frac{\partial P_{T}}{\partial P_{0}}\right]\\
# = & \int_{-\infty}^{\infty}W_{0}^{-\gamma}\left[n_{s}e^{(d-r)T}+n_{o}e^{-rT}\mathbb{I}_{P_{T}>K}\right]\exp\left\{ \left(r-d-\frac{\sigma^{2}}{2}\right)T+u\sigma\sqrt{T}\right\} f(u)du
# \end{align*}

# In[ ]:

def f_UPPS_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K):
    """
    Utility-adjusted PPS (UPPS)
    The incentive constraint of the agent (UPPS).
    """
    W_T = f_W_T_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K)
    value = pow(W_T, -gamma) * f_W_T_to_P_T_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K) * f_P_T_to_P_0(v, r_f, d, s, T)
    return value

def f_UPPS_integral_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K):
    """
    UPPS integral.
    """
    func = f_UPPS_pc(v, P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K) * norm.pdf(v, eta_m, eta_s)
    return func

def f_E_UPPS_pc(variables_pc, parameters_pc):
    """
    Expected utility-adjusted PPS.
    """
    phi, n_s, n_o = variables_pc
    P_0, r_f, d, s, T, wealth, K = parameters_pc
    value = integrate.quad(f_UPPS_integral_pc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, phi, n_s, n_o, K))[0]
    return value


# ### General contract
# 

# In[ ]:

def f_UPPS_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1):
    """
    Utility-adjusted PPS (UPPS)
    The incentive constraint of the agent (UPPS).
    """
    ln_P_T = f_ln_P_T(v, P_0, r_f, d, s, T)
    
    (beta_0, beta_1, beta_2) = f_beta(P_0, r_f, d, s, T)
    
    W_T = f_W_T_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1)
    W_T_to_P_T = f_W_T_to_P_T_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1)
    P_T_to_P_0 = f_P_T_to_P_0(v, r_f, d, s, T)
    
    value = math.pow(W_T, -gamma) * W_T_to_P_T * P_T_to_P_0
    return value

def f_UPPS_integral_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1):
    """
    UD integral.
    """
    func = f_UPPS_gc(v, P_0, r_f, d, s, T, wealth, alpha_0, alpha_1) * norm.pdf(v, eta_m, eta_s)
    return func

def f_E_UPPS_gc(variables_gc, parameters_gc):
    """
    Expected utility-adjusted PPS.
    """
    alpha_0, alpha_1 = variables_gc
    P_0, r_f, d, s, T, wealth, K = parameters_gc
    value = integrate.quad(f_UPPS_integral_gc, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                           args=(P_0, r_f, d, s, T, wealth, alpha_0, alpha_1))[0]
    return value


# # 3. Implement the functions

# ## Define distance metrics

# In[ ]:

def f_distance(u, P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt, K_opt):
    W_0 = f_W_0(u, P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth)
    W_0_opt = f_W_0(u, P_0, r_f, d, sigma, T, phi_opt, n_s_opt, n_o_opt, K_opt, wealth)
    value = abs(W_0 - W_0_opt) / W_0
    return value

def f_distance_integral(u, P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt, K_opt):
    func = f_distance(u, P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt, K_opt) * norm.pdf(u)
    return func

def f_E_distance(P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt, K_opt):
    value = integrate.quad(f_distance_integral, -10, 10,
                           epsabs=1e-14, epsrel=1e-14, limit=100,
                          args=(P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt, K_opt))[0]
    return value


# ### Optimal contract

# ### Piece-wise linear contract with 2 constraints

# Test

# In[ ]:

df = pd.read_csv("representative_contract.csv")
representative_contract = df.values.tolist()


# In[ ]:

representative_contract[0] = [0.0, 100.0, 0.0435, 0.0269, 0.3315, 4.3122, 0.0941, 0.0029, 0.0089, 88.254, 1.1465, 11.091]


# In[ ]:

P_0, r_f, d, s, T, phi, n_s, n_o, K, wealth, rescale_factor = representative_contract[0][1:]
print('The original contract is:')
print('P_0 = {}'.format(P_0))
print('s = {}'.format(s))
print('T = {}'.format(T))
print('phi = {}'.format(phi))
print('n_s = {}'.format(n_s))
print('n_o = {}'.format(n_o))
print('K = {}'.format(K))
print('wealth = {}'.format(wealth))
print('rescale factor = {}'.format(rescale_factor))

variables_pc = [phi, n_s, n_o]
parameters = [P_0, r_f, d, s, T, wealth, K]

E_U = f_E_U_pc(variables_pc, parameters)
print('EU: {}'.format(E_U))
E_UPPS = f_E_UPPS_pc(variables_pc, parameters)
print('EUPPS: {}'.format(E_UPPS))
E_cost_0 = f_E_cost_0_pc(variables_pc, parameters)
print('Ecost: {}m'.format(E_cost_0 * rescale_factor))


# ### Optimal piecewise linear contract

# In[ ]:

cons = ({'type': 'eq', 
         'fun': lambda variables, parameters: f_E_U_pc(variables, parameters) - E_U, 
         'args': (parameters,)},
        {'type': 'eq',
         'fun': lambda variables, parameters: f_E_UPPS_pc(variables, parameters) - E_UPPS, 
         'args': (parameters,)}
       )

bnds = ((-wealth, np.inf), (0.00, 1.00), (-np.inf, np.inf))

print('\nOptimal piecewise linear contract')
contract_opt = minimize(fun=f_E_cost_T_pc, x0=variables_pc, args=parameters, method='SLSQP',
                        bounds=bnds, constraints=cons, tol=None, callback=None, 
                        options={'disp': False, 'iprint': 1, 'eps': 1.4901161193847656e-08, 'maxiter': 100, 'ftol': 1e-06})
phi_opt, n_s_opt, n_o_opt = contract_opt.x
print('phi = {}'.format(phi_opt))
print('n_s = {}'.format(n_s_opt))
print('n_o = {}'.format(n_o_opt))        
variables_opt = [phi_opt, n_s_opt, n_o_opt]
E_U_opt = f_E_U_pc(variables_opt, parameters)
print('EU: {}'.format(E_U_opt))
E_UPPS_opt = f_E_UPPS_pc(variables_opt, parameters)
print('EUPPS: {}'.format(E_UPPS_opt))
E_cost_0_opt = f_E_cost_0_pc(variables_opt, parameters)
print('Ecost: {}m'.format(E_cost_0_opt * rescale_factor))


# ### Optimal general contract

# In[ ]:

cons = ({'type': 'ineq', 
         'fun': lambda variables, parameters: f_E_U_gc(variables, parameters) - E_U, 
         'args': (parameters,)},
        {'type': 'eq',
         'fun': lambda variables, parameters: f_E_UPPS_gc(variables, parameters) - E_UPPS, 
         'args': (parameters,)}
       )

bnds = ((-np.inf, np.inf), (0, np.inf))

print('\nOptimal piecewise linear contract')
print(f_beta(P_0, r_f, d, s, T))
contract_opt = minimize(fun=f_E_cost_T_gc, x0=[-2, 1], args=parameters, method='SLSQP',
                        bounds=bnds, constraints=cons, tol=None, callback=None, 
                        options={'disp': True, 'iprint': 2, 'eps': 1.4901161193847656e-08, 'maxiter': 100, 'ftol': 1e-06})
alpha_0_opt, alpha_1_opt = list(contract_opt.x)
print('alpha_0 = {}'.format(alpha_0_opt))
print('alpha_1 = {}'.format(alpha_1_opt)) 
variables_opt = [alpha_0_opt, alpha_1_opt]
E_U_opt = f_E_U_gc(variables_opt, parameters)
print('EU: {}'.format(E_U_opt))
E_UPPS_opt = f_E_UPPS_gc(variables_opt, parameters)
print('EUPPS: {}'.format(E_UPPS_opt))
E_cost_0_opt = f_E_cost_0_gc(variables_opt, parameters)
print('Ecost: {}m'.format(E_cost_0_opt * rescale_factor))


# Loops

# In[ ]:

# for gamma in [0.5, 1.000001, 2, 3, 4, 5, 6]:
#     print('gamma = {}\n'.format(gamma))
#     list_optimal_contract = []
#     for count in range(0, len(list_representative_contract)):
#         representative_contract = list_representative_contract[count]
#     #     print('The contract is {}'.format(representative_contract))
#         try:
#             P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, rescale_factor = representative_contract
#             print('The original contract is:')
#             print('sigma = {}'.format(sigma))
#             print('T = {}'.format(T))
#             print('phi = {}'.format(phi))
#             print('n_s = {}'.format(n_s))
#             print('n_o = {}'.format(n_o))
#             print('K = {}'.format(K))
#             print('wealth = {}'.format(wealth))
#             print('rescale factor = {}'.format(rescale_factor))

#             variables = [phi, n_s, n_o]
#             parameters = [P_0, r_f, d, sigma, K, wealth, gamma, T]

#             E_U = f_E_U(variables, parameters)
#             print('EU: {}'.format(E_U))
#             E_UD = f_E_UD(variables, parameters)
#             print('EUD: {}'.format(E_UD))
#             E_UT = f_E_UT(variables, parameters)
#             print('EUT: {}'.format(E_UT))
#             E_cost = f_E_cost(variables, parameters)
#             print('Ecost: {}m'.format(E_cost * rescale_factor))

#             cons = ({'type': 'eq',
#                      'fun': lambda variables, parameters: f_E_UD(variables, parameters) - E_UD, 
#                      'args': (parameters,)},
#                     {'type': 'ineq', 
#                      'fun': lambda variables, parameters: f_E_U(variables, parameters) - E_U, 
#                      'args': (parameters,)}
#                    )

#             bnds = ((-wealth, np.inf), (0.00, 1.00), (0.00, np.inf))

#             print('\nOptimal piecewise linear contract')
#             contract_opt = list(minimize(f_E_cost, variables, args=parameters, method='SLSQP', bounds=bnds, constraints=cons).x)
#             phi_opt, n_s_opt, n_o_opt = contract_opt
#             print('phi = {}'.format(phi_opt))
#             print('n_s = {}'.format(n_s_opt))
#             print('n_o = {}'.format(n_o_opt))        
#             variables_opt = contract_opt
#             E_U_opt = f_E_U(variables_opt, parameters)
#             print('EU: {}'.format(E_U))
#             E_UD_opt = f_E_UD(variables_opt, parameters)
#             print('EUD: {}'.format(E_UD))
#             E_UT_opt = f_E_UT(variables_opt, parameters)
#             print('EUT: {}'.format(E_UT_opt))
#             E_cost_opt = f_E_cost(variables_opt, parameters)
#             print('Ecost: {}m'.format(E_cost_opt * rescale_factor))

#             distance = f_E_distance(P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt)
#             savings = abs(E_cost - E_cost_opt) / E_cost
#             print('distance = {}'.format(distance))
#             print('savings = {}'.format(savings))
#             print('')
#             print('-------------------------------------------------------------------')

#             result = [phi_opt, n_s_opt, n_o_opt, E_cost * rescale_factor, E_cost_opt * rescale_factor, distance, savings]
#             list_optimal_contract.append(result)
#         except:
#             result = [0, 0, 0, 0, 0, 0, 0]
#             list_optimal_contract.append(result)
#             print('Cannot solve it!')
#             print('')

#     # Put the original contract and the optimal contract together and output the file.
#     result = []
#     for i in range(len(list_optimal_contract)):
#         a = list_representative_contract[i] + list_optimal_contract[i]
#         result.append(a)

#     df = pd.DataFrame(result)
#     df.columns = ['P_0', 'r_f', 'd', 'sigma', 'T', 'phi', 'n_s', 'n_o', 'K', 'wealth', 'rescale_factor', 'phi_opt', 'n_s_opt', 'n_o_opt', 'E_cost', 'E_cost_opt', 'distance', 'savings']
#     df.to_csv('result_piecewise_2_constraints_gamma_{}.csv'.format(gamma))

