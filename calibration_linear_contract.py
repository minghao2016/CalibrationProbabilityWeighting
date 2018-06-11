"""
Purpose: Calibration of the project "probability weighting CEOs"

Author: Yuhao Zhu

Versions: See readme file.
"""

import sys, os
sys.path.append(os.getcwd())

from include.functions import *
from scipy.optimize import minimize
import csv
import pandas as pd


df = pd.read_csv("data/representative_contracts.csv")
list_representative_contracts = df.values.tolist()
# print(df.head())

"""
Calibration using approximated theta.
"""

"""
abs_diff = [
#    (0.1, 21.1123266358844, 60.645676606364106),
#    (0.2, 8.651929204945983, 13.524499495663248),
    (0.3, 5.074542788364155, 4.829340618286907),
    (0.4, 3.39491848978483, 1.9998947379100558),
    (0.5, 2.42854340608065, 0.8671730118532825),
    (0.6, 1.856646898864211, 0.3788550913396194),
    (0.7, 1.5028718106121404, 0.15596800404500352),
    (0.8, 1.2709159554376692, 0.053362177417372515),
    (0.9, 1.1120072148842672, 0.010666897104624415)
]

square_diff = [
#    (0.1, 1.3730384415052759, 5.609934733205399),
#    (0.2, 5.490607798404917, 8.407196974235255),
    (0.3, 3.8611937871408837, 3.582617741605922),
    (0.4, 2.884768181778572, 1.657024470484358),
    (0.5, 2.233070013091103, 0.7790848676727394),
    (0.6, 1.787746146210776, 0.357638829839582),
    (0.7, 1.478951570393733, 0.15138856920931862),
    (0.8, 1.26259402734772, 0.05262966657816373),
    (0.9, 1.1095263507982236, 0.010608602819599581)
]

for gamma in [0.2]:
    for approximates in [square_diff[-1]]:
        (delta, eta_s, eta_m) = approximates
        print('gamma {} delta {} begins!'.format(gamma, delta))
        list_optimal_contracts = []
        for count in range(0, len(list_representative_contracts)):
            print('Start the contract {}'.format(count))
            try:
                co_per_rol, P_0, r_f, d, s, T, phi, n_s, n_o, K, wealth, rescale_factor = list_representative_contracts[count]

                variables_pc = [phi, n_s, n_o]
                parameters = [P_0, r_f, d, s, T, wealth, K]

                E_U = f_E_U_pc(variables_pc, parameters, gamma, eta_s, eta_m, wealth_min=0.01)
                E_UPPS = f_E_UPPS_pc(variables_pc, parameters, gamma, eta_s, eta_m, wealth_min=0.01)
                E_cost_0 = f_E_cost_0_pc(variables_pc, parameters)

                #############################################################################################
                # Optimization program
                #############################################################################################

                cons = ({'type': 'ineq', 
                         'fun': lambda variables, parameters: f_E_U_pc(variables, parameters, gamma, eta_s, eta_m, wealth_min=0.01) - E_U, 
                         'args': (parameters,)},
                        {'type': 'ineq',
                         'fun': lambda variables, parameters: f_E_UPPS_pc(variables, parameters, gamma, eta_s, eta_m, wealth_min=0.01) - E_UPPS, 
                         'args': (parameters,)}
                       )

                bnds = ((-wealth, np.inf), (0.00, 1.00), (0, 1.00))

                contract_opt = minimize(fun=f_E_cost_T_pc, x0=variables_pc, args=parameters, method='SLSQP',
                                        bounds=bnds, constraints=cons, tol=None, callback=None, 
                                        options={'disp': False, 'iprint': 1, 'eps': 1.4901161193847656e-08, 'maxiter': 100, 'ftol': 1e-06})
                phi_opt, n_s_opt, n_o_opt = contract_opt.x
                variables_opt = [phi_opt, n_s_opt, n_o_opt]
                E_U_opt = f_E_U_pc(variables_opt, parameters, gamma, eta_s, eta_m, wealth_min=0.01)
                E_UPPS_opt = f_E_UPPS_pc(variables_opt, parameters, gamma, eta_s, eta_m, wealth_min=0.01)
                E_cost_0_opt = f_E_cost_0_pc(variables_opt, parameters)

                distance = f_E_distance(P_0, r_f, d, s, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt)
                savings = (E_cost_0 - E_cost_0_opt) / E_cost_0
                arguments = [phi_opt, n_s_opt, n_o_opt]
                statistics = [E_U, E_UPPS, E_cost_0 * rescale_factor, E_U_opt, E_UPPS_opt, E_cost_0_opt * rescale_factor]
                flags = [distance, savings]
                results = arguments + statistics + flags
                list_optimal_contracts.append(results)
            except:
                results = [0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0]
                list_optimal_contracts.append(results)

        # Put the original contract and the optimal contract together and output the file.
        results = []
        for i in range(len(list_optimal_contracts)):
            whole_contract = list_representative_contracts[i] + list_optimal_contracts[i]
            results.append(whole_contract)

        df = pd.DataFrame(results)
        df.columns = ['co_per_rol', 'P_0', 'r_f', 'd', 'sigma', 'T', 'phi', 'n_s', 'n_o', 'K', 'wealth', 'rescale_factor', 
                      'phi_opt', 'n_s_opt', 'n_o_opt', 
                      'E_U', 'E_UPPS', 'E_cost_0', 'E_U_opt', 'E_UPPS_opt', 'E_cost_0_opt',
                      'distance', 'savings']
        df.to_csv('results_piecewise_contracts/gamma_{}_delta_{}.csv'.format(gamma, delta))
        print('File results_piecewise_contracts/gamma_{}_delta_{}.csv saved!'.format(gamma, delta))

"""

"""
Calibration using eta_s and eta_m.
"""
# gamma = 1.01, 2, 3, 5, 8

for eta_m in [0]:
    for gamma in [1.01, 2, 3, 5, 8]:
        for eta_s in [1.25]:
            print('=' * 60)
            print('gamma {} eta_s {} eta_m {} begins!'.format(gamma, eta_s, eta_m))
            list_optimal_contracts = []
            for count in range(0, len(list_representative_contracts)):
                print('Start the contract {}'.format(count))
                try:
                    co_per_rol, P_0, r_f, d, s, T, phi, n_s, n_o, K, wealth, rescale_factor = list_representative_contracts[count]

                    variables_pc = [phi, n_s, n_o]
                    parameters = [P_0, r_f, d, s, T, wealth, K]

                    E_U = f_E_U_pc(variables_pc, parameters, gamma, eta_s, eta_m, wealth_min=0.01)
                    E_UPPS = f_E_UPPS_pc(variables_pc, parameters, gamma, eta_s, eta_m, wealth_min=0.01)
                    E_cost_0 = f_E_cost_0_pc(variables_pc, parameters)

                    # Optimization program
            
                    cons = ({'type': 'ineq', 
                            'fun': lambda variables, parameters: f_E_U_pc(variables, parameters, gamma, eta_s, eta_m, wealth_min=0.01) - E_U, 
                            'args': (parameters,)},
                            {'type': 'ineq',
                            'fun': lambda variables, parameters: f_E_UPPS_pc(variables, parameters, gamma, eta_s, eta_m, wealth_min=0.01) - E_UPPS, 
                            'args': (parameters,)}
                        )

                    bnds = ((-wealth, np.inf), (0.00, 1.00), (0, 1.00))

                    contract_opt = minimize(fun=f_E_cost_T_pc, x0=variables_pc, args=parameters, method='SLSQP',
                                            bounds=bnds, constraints=cons, tol=None, callback=None, 
                                            options={'disp': False, 'iprint': 1, 'eps': 1.4901161193847656e-08, 'maxiter': 100, 'ftol': 1e-06})
                    phi_opt, n_s_opt, n_o_opt = contract_opt.x
                    variables_opt = [phi_opt, n_s_opt, n_o_opt]
                    E_U_opt = f_E_U_pc(variables_opt, parameters, gamma, eta_s, eta_m, wealth_min=0.01)
                    E_UPPS_opt = f_E_UPPS_pc(variables_opt, parameters, gamma, eta_s, eta_m, wealth_min=0.01)
                    E_cost_0_opt = f_E_cost_0_pc(variables_opt, parameters)

                    # Distance metrics
                    distance = f_E_distance(P_0, r_f, d, s, T, phi, n_s, n_o, K, wealth, phi_opt, n_s_opt, n_o_opt)
                    savings = (E_cost_0 - E_cost_0_opt) / E_cost_0
                    arguments = [phi_opt, n_s_opt, n_o_opt]
                    statistics = [E_U, E_UPPS, E_cost_0 * rescale_factor, E_U_opt, E_UPPS_opt, E_cost_0_opt * rescale_factor]
                    flags = [distance, savings]
                    results = arguments + statistics + flags
                    list_optimal_contracts.append(results)
                except:
                    results = [0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0]
                    list_optimal_contracts.append(results)

            # Put the original contract and the optimal contract together and output the file.
            results = []
            for i in range(len(list_optimal_contracts)):
                whole_contract = list_representative_contracts[i] + list_optimal_contracts[i]
                results.append(whole_contract)

            df = pd.DataFrame(results)
            df.columns = ['co_per_rol', 'P_0', 'r_f', 'd', 'sigma', 'T', 'phi', 'n_s', 'n_o', 'K', 'wealth', 'rescale_factor', 
                        'phi_opt', 'n_s_opt', 'n_o_opt', 
                        'E_U', 'E_UPPS', 'E_cost_0', 'E_U_opt', 'E_UPPS_opt', 'E_cost_0_opt',
                        'distance', 'savings']
            df.to_csv('results/gamma_{}_eta_s_{}_eta_m_{}.csv'.format(gamma, eta_s, eta_m))
            print('File gamma_{}_eta_s_{}_eta_m_{}.csv saved!'.format(gamma, eta_s, eta_m))
            print('')
