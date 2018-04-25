#%%

import matplotlib.pyplot as plt
import math
import numpy as np

# Draw figures

# In[2]:

def f_W_0(u, P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth_0):
    """
    The wage payment to the agent given the stock price discounted to present.
    """
    sigma_2 = pow(sigma, 2)
    P_T = f_P_T(u, P_0, r_f, d, sigma, T)
    W_0 = (phi + wealth_0) + n_s * math.exp((d - r_f) * T) * P_T + n_o * math.exp(-r_f * T) * max(P_T - K, 0)
    return W_0

def f_W_0_opt(u, P_0, r_f, d, sigma, T, alpha_0, alpha_1, alpha_2, gamma):
    """
    The wage payment (optimal) to the agent given the stock price discounted to present.
    """
    P_T = f_P_T(u, P_0, r_f, d, sigma, T)
    ln_P_T = f_ln_P_T(u, P_0, r_f, d, sigma, T)
    if ln_P_T > - alpha_1 / (2 * alpha_2):
        value = alpha_0 + alpha_1 * ln_P_T + alpha_2 * pow(ln_P_T, 2)
    else:
        value = alpha_0 - pow(alpha_1, 2) / (4 * alpha_2)
    expression = max(value, pow(0.0001, gamma))
    W_0 = pow(expression, 1 / gamma)
    return W_0


# In[24]:

# Observed contract
P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth_0, rescale_factor = [100.0, 0.0435, 0.0269, 0.3315, 4.3122,
                                                                     0.0941, 0.0029, 0.0089, 88.254, 1.1465, 11.091]


# In[25]:

x = np.linspace(0,300,10000)
y = []
for P_T in x:
    W_0 = phi + n_s * math.exp((d - r_f) * T) * P_T + n_o * math.exp(-r_f * T) * max(P_T - K, 0)
    W_0 = W_0 * rescale_factor
    y.append(W_0)


# In[26]:

# optimal pw
P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth_0, rescale_factor = [100.0, 0.0435, 0.0269, 0.3315, 4.3122,
                                                                     0.05888707058632977, 0.00503489078157908, 0.002137273775218311,
                                                                     88.254, 1.1465, 11.091]


# In[27]:

x_pw = np.linspace(0,300,10000)
y_pw = []
for P_T in x_pw:
    W_0 = phi + n_s * math.exp((d - r_f) * T) * P_T + n_o * math.exp(-r_f * T) * max(P_T - K, 0)
    W_0 = W_0 * rescale_factor
    y_pw.append(W_0)


# In[28]:

# optimal crra
P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth_0, rescale_factor = [100.0, 0.0435, 0.0269, 0.3315, 4.3122,
                                                                     -0.2990542090748607, 0.012114849391333877, -0.012383702206631363,
                                                                     88.254, 1.1465, 11.091]


# In[29]:

x_crra = np.linspace(0,300,10000)
y_crra = []
for P_T in x_crra:
    W_0 = phi + n_s * math.exp((d - r_f) * T) * P_T + n_o * math.exp(-r_f * T) * max(P_T - K, 0)
    W_0 = W_0 * rescale_factor
    y_crra.append(W_0)


# In[32]:

plt.plot(x, y, 'g', label='Observed contract')
plt.plot(x_crra, y_crra, 'r--', label='Optimal contract with CRRA')
plt.plot(x_pw, y_pw, 'b--', label='Optimal contract with pw')
plt.xlabel("$P_T/P_0 * 100$")
plt.ylabel("Cost of the firm in million dollars at $T=0$")
plt.legend(loc='lower right')
plt.savefig('figure/observed_and_calculated.jpg', dpi=720, format='jpg')


# In[72]:

P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth_0, rescale_factor = [100.0, 0.0173, 0.029999999, 0.22458053, 6.90218914074764, 0.014607919, 7.8355602e-05, 0.005904167911500001, 126.91431038656421, 0.071425304, 390.56683]
alpha_0, alpha_1, alpha_2 = [0.016757797295997037, -0.010760350120053903, 0.0017339533651527439]
gamma = 3
x_opt = np.linspace(0.0001,200,10000)
y_opt = []
for P_T in x_opt:
    ln_P_T = math.log(P_T)
    if ln_P_T > - alpha_1 / (2 * alpha_2):
        value = alpha_0 + alpha_1 * ln_P_T + alpha_2 * pow(ln_P_T, 2)
    else:
        value = alpha_0 - pow(alpha_1, 2) / (4 * alpha_2)
    expression = max(value, pow(0.001, gamma))
    W_0 = (pow(expression, 1 / gamma) - wealth_0) * rescale_factor
    y_opt.append(W_0)


# In[82]:

plt.plot(x, y, label='Observed contract')
plt.plot(x_opt, y_opt, 'r--', label='Optimal contract')
plt.xlabel("$P_T$ when $P_0 = 100$")
plt.ylabel("Cost of the firm in million dollars")
plt.legend()
plt.show()


# In[35]:

P_0, r_f, d, sigma, T, phi, n_s, n_o, K, wealth_0, rescale_factor = [100.0, 0, 0, 0.25, 10, 1, 0.01, 0.05, 100, 1, 100]
alpha_0, alpha_1, alpha_2 = [10, -4, 2]
gamma = 3
x_opt = np.linspace(0.0001,15,10000)
y_opt = []
for P_T in x_opt:
    ln_P_T = math.log(P_T)
    if ln_P_T > - alpha_1 / (2 * alpha_2):
        value = alpha_0 + alpha_1 * ln_P_T + alpha_2 * pow(ln_P_T, 2)
    else:
        value = alpha_0 - pow(alpha_1, 2) / (4 * alpha_2)
    expression = max(value, pow(0.001, gamma))
    W_0 = (pow(expression, 1 / gamma) - wealth_0)
    y_opt.append(W_0)


# In[36]:

plt.plot(x_opt, y_opt, 'r--', label='Optimal contract')
plt.xlabel("$P_T$")
plt.ylabel("$W_0$")
plt.legend()
plt.show()


# # Comparison of optimal contracts

# In[52]:

r = np.linspace(0, 20, 10001)[1:]
w_pw = []
w_rti = []
w_crra = []
for x in r:
    try:
        y = (0.75*math.exp(0.5+0.111*math.log(x)+0.5*math.log(x)**2)*(1+0.5*math.log(x)))**0.3
        if isinstance(y, complex):
            y = 0
        w_pw.append(y)    
    except:
        w_pw.append(0)
        
    try:
        w_rti.append((0.75*(1+0.5*math.log(x)+2*math.log(x)**2))**0.5)
    except:
        w_rti.append(0)
        
    try:
        y = (1*(3+3*math.log(x)))**0.3
        if isinstance(y, complex):
            y = 0
        w_crra.append(y)    
    except:
        w_crra.append(0)
min_w_rti = min(w_rti)
for x in range(np.argmin(w_rti)):
     w_rti[x] = min_w_rti


# In[56]:

plt.plot(r, w_pw, '-', color='b', label='probability weighting model')
plt.plot(r, w_crra, '-.', color='r', label='CRRA model')
plt.xlabel('Firm performance')
plt.ylabel('CEO wealth')
plt.legend(loc='lower right', prop={'size': 10})
plt.xlim([0, 10])
plt.ylim([0, 4])
plt.title('Comparison: shapes of contracts in different models')
plt.savefig('figure/comparison_models.jpg', dpi=720, format='jpg')


# In[22]:

plt.plot(r, w_pw, '-', color='b', label='probability weighting')
plt.plot(r, w_rti, '--', color='g', label='risk-taking incentive')
plt.plot(r, w_crra, '-.', color='r', label='traditional model')
plt.xlabel('Firm performance')
plt.ylabel('CEO wealth')
plt.legend(loc='lower right', prop={'size': 10})
plt.xlim([0, 8])
plt.ylim([0, 4])
plt.title('Comparison of different models')
plt.savefig('figure/comparison_models_20170808.jpg', dpi=720, format='jpg')