# CalibrationProbabilityWeighting

Calibration with probability weighting model for CEO contracts.

Purpose: Calibration of the project "probability weighting CEOs"

Author: Yuhao Zhu

Versions:
    20170226: Create the function.
    20170227: Create the test.
    20170310: Improvement. Can be used to calculate the expected profit.
    20170327: Re-write the code. No dictionary any more. Can now search for best contract without constraints.
    20170419: Merge the codes into a notebook.
    20170421: Write instructions for calibration and write instructions for calculating representative contract.
    20170422: Write the code for calculating the representative contract.
            The maturity time must be bounded between 0.1 to 20 to ensure a valid integration.
    20170516: Minor changes. Use simpler variables names.
    20170524: Change the random variable from P_T to u. Use normal distribution rather than log-normal distribution.
    20170525: Changes.
    20170601: Changes.
    20170602: Changes.
    20170603: Write the code to search for the optimal contracts given constraints including Theta.
    20170604: Changes.
    20170605: Changes.
    20170610: Changes.
    20170726: I begin to modify the codes for probability weighting. Make representative contract an extra file.
    20170802: Changes.
    20170804: Refine the algorithm. Use v as the pdf to avoid complex computation.
            Also set eta_s, eta_m, and gamma as global variables. This reduce the complexity of the functions.
    20170807: Changes.
    20180320: Remove all contents about general optimal contract. Then create a new file containing these codes. "calibration_probability_weighting_general_contract.ipynb".
    20180425: Migrate from ipynb to Python. Write a seperate file for functions.
    20180426: Calibrate for different eta_s.