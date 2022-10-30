from QMLE_scipy import *
import random as random
from tabulate import tabulate
from sklearn.metrics import mean_squared_error
from math import sqrt

# Authors: Authors: Agostino Capponi, Mohammadreza Bolandnazar, Erica Zhang
# License: MIT License
# Version: Oct 18, 2022

# DESCRIPTION: This package provides method to generate time-varying row-normalized weight matrices to explore QMLE estimators. It also provides monte carlo simulation as well as statistics table to explore the performance of QMLE estimators.

def is_even(num):    
    r"""test is a number is even
     
    The main idea of this function is to determine if a given integer is even. 
    
    Parameters
    ----------
    num : int
        input number.
        
    Returns
    -------
    Bool
        True if even; False if not.
     """
    if (num % 2) == 0:
        return True
    else:
        return False


def generate_weight_matrix(n):
    r"""generate simple weight matrix
     
    The main idea of this function is to generate simple row-normalized weight matrix.
    
    Parameters
    ----------
    n: int
        Determine that the weight matrix is n-by-n.
        
    Returns
    -------
    ndarray
        returns a row-normalized n-by-n weight matrix.
     """
    row = []
    for i in range(n):
        temp_row = np.zeros(n)
        # ensure that diagonal is nonzero and that the row is normalized
        no_stop = True
        while no_stop:
            place_one = random.randint(0,n-1)
            if place_one != i:
                no_stop = False
        temp_row[place_one] = 1
        row.append(temp_row.tolist())
    w = np.array(row).reshape(n,n)
    return w



def generate_wls(n,T, alternate = True):
    r"""generate simple weight matrix list
     
    The main idea of this function is to generate a list of simple row-normalized weight matrices.
    
    Parameters
    ----------
    n : int
        Determine that the weight matrix is n-by-n.
    T : int
        Determine the number of time points.
    alternate : Bool
        True if requires weight matrix list to alternate at even and odd time points.
        
    Returns
    -------
    list; ndarray
        returns a list of row-normalized n-by-n weight matrices.
     """
    
    w0 = generate_weight_matrix(n)
    w1 = generate_weight_matrix(n)
    W_ls = []
    if alternate:
        for i in range(T+1):
            if is_even(i):
                W_ls.append(w0)
            else:
                W_ls.append(w1) 
    else:
        for i in range(T+1):
            W_ls.append(w0)
    return W_ls 


def generate_samples_scipy_fix_weight(n,T,t_theta,g_theta, W_ls, N=50, constrain=True):
    r"""generate list of estimators from monte-carlo experiments with fixed weight matrices
     
    The main idea of this function is to generate a list of returned estimators from multiple monte-carlo experiments with weight matrices fixed for each experiment.
    
    Parameters
    ----------
    n : int
        Determine that the weight matrix is n-by-n.
    T : int
        Determine the number of time points.
    t_theta : list; float64
        List of true theta in order of sig, lam, gamma, rho, beta.
    g_theta : list; float64
        List of guessed theta in order of sig, lam, gamma, rho, beta.
    W_ls : list; ndarray; float64
        List of weight matrices, fixed for each monte-carlo experiment.
    N : int
        Number of monte-carlo experiments.
    constrain : bool
        Select if the chosen optimization is constrained or not.
                
    Returns
    -------
    list; object
        returns a list of the scipy_res objects.
     """
    # set up true and guess parameters
    t_sig, t_lam, t_gamma, t_rho = t_theta[:4]
    t_beta = t_theta[4:]
    g_sig, g_lam, g_gamma, g_rho = g_theta[:4]
    g_beta = g_theta[4:]
    k = len(g_beta)
    # set up initial guess
    initial_guess = g_theta
    # sample list
    sample_ls = []
    for r in range(N):
        # set up samples
        alpha = np.random.normal(0,1,T).reshape(n,1)
        # avoid extreme values
        alpha = np.nan_to_num(alpha)
        alpha[alpha == -np.inf] = 0
        alpha[alpha == np.inf] = 0
        # avoid extreme values
        c0 = np.random.normal(0,1,n).reshape(n,1)
        c0 = np.nan_to_num(c0)
        c0[c0 == -np.inf] = 0
        c0[c0 == np.inf] = 0
        
        x = []
        for i in range(T):
            tem = np.random.normal(0,1,n*k).reshape(n,k)
            # avoid extreme values
            tem = np.nan_to_num(tem)
            tem[tem == -np.inf] = 0
            tem[tem == np.inf]=0
            x.append(tem)
            
        V_nt = []
        for i in range(T):
            tem = np.random.normal(0,t_sig,n).reshape(n,1)
            # avoid extreme values
            tem = np.nan_to_num(tem)
            tem[tem == -np.inf] = 0
            tem[tem == np.inf]=0
            V_nt.append(tem)
        
        Y0 = np.random.normal(0,1,n).reshape(n,1)
        Y_ls = []
        Y_ls.append(Y0)
        
        for i in range(T):
            l_n = np.ones(n).reshape(n,1)
            c_vec = t_gamma*Y_ls[i]+t_rho*np.matmul(W_ls[i],Y_ls[i])+np.matmul(x[i],np.array(t_beta).reshape(k,1)).reshape(n,1)+c0+alpha[i]*l_n+V_nt[i]
            Y_nt = np.matmul(np.linalg.inv(np.identity(n)-t_lam*W_ls[i+1]),c_vec)
            Y_ls.append(Y_nt)
        
        # run scipy optimize
        sample = QMLE_scipy_estimate(x, Y_ls, W_ls, initial_guess, constrain).params
        sample_ls.append(sample)
        
    # return sample list
    return sample_ls


def generate_samples_scipy_flex_weight(n,T, t_theta,g_theta, N=500, constrain=True):
    r"""generate list of estimators from monte-carlo experiments with changing weight matrices
     
    The main idea of this function is to generate a list of returned estimators from multiple monte-carlo experiments without fixing weight matrices for each experiment.
    
    Parameters
    ----------
    n : int
        Total number of items at each time point.
    T : int
        Determine the number of time points.
    t_theta : list; float64
        List of true theta in order of sig, lam, gamma, rho, beta.
    g_theta : list; float64
        List of guessed theta in order of sig, lam, gamma, rho, beta.
    N : int
        Number of monte-carlo experiments.
    constrain : bool
        Select if the chosen optimization is constrained or not.
                
    Returns
    -------
    list; list; float64
        returns a list of estimated parameter list for each experiment.
     """
    # set up true and guess parameters
    t_sig, t_lam, t_gamma, t_rho = t_theta[:4]
    t_beta = t_theta[4:]
    g_sig, g_lam, g_gamma, g_rho = g_theta[:4]
    g_beta = g_theta[4:]
    k = len(g_beta)
    # set up initial guess
    initial_guess = g_theta
    
    sample_ls = []
    for r in range(N):
        # set up samples
        alpha = np.random.normal(0,1,T).reshape(n,1)
        alpha = np.nan_to_num(alpha)
        alpha[alpha == -np.inf] = 0
        alpha[alpha == np.inf] = 0
        c0 = np.random.normal(0,1,n).reshape(n,1)
        c0 = np.nan_to_num(c0)
        c0[c0 == -np.inf] = 0
        c0[c0 == np.inf] = 0
        x = []
        for i in range(T):
            tem = np.random.normal(0,1,n*k).reshape(n,k)
            # avoid extreme values
            tem = np.nan_to_num(tem)
            tem[tem == -np.inf] = 0
            tem[tem == np.inf]=0
            x.append(tem)
        # manually setting up a row-normalized spatial weight vector with 0 diagonals for even time points
        w0 = generate_weight_matrix(n)
        w1 = generate_weight_matrix(n)
        W_ls = []
        for i in range(T+1):
            if is_even(i):
                W_ls.append(w0)
            else:
                W_ls.append(w1)
                
        V_nt = []
        for i in range(T):
            tem = np.random.normal(0,t_sig,n).reshape(n,1)
            # avoid extreme values
            tem = np.nan_to_num(tem)
            tem[tem == -np.inf] = 0
            tem[tem == np.inf]=0
            V_nt.append(tem)
            
        Y0 = np.random.normal(0,1,n).reshape(n,1)
        Y_ls = []
        Y_ls.append(Y0)
        
        for i in range(T):
            l_n = np.ones(n).reshape(n,1)
            c_vec = t_gamma*Y_ls[i]+t_rho*np.matmul(W_ls[i],Y_ls[i])+np.matmul(x[i],np.array(t_beta).reshape(k,1)).reshape(n,1)+c0+alpha[i]*l_n+V_nt[i]
            Y_nt = np.matmul(np.linalg.inv(np.identity(n)-t_lam*W_ls[i+1]),c_vec)
            Y_ls.append(Y_nt)
    
        # run scipy optimize
        sample = QMLE_scipy_estimate(x, Y_ls, W_ls, initial_guess, constrain).params
        sample_ls.append(sample)
    # return sample list
    return sample_ls



def obtain_table_stats(sample_ls,t_theta,T):
    r"""generate table of stats for the performance of estimators from monte-carlo experiments
     
    The main idea of this function is to generate a table of stats for the performance of estimators from monte-carlo experiments.
    
    Parameters
    ----------
    sample_ls: list; list; float64
        A list of estimated parameter list for each monte-carlo experiment.
    t_theta : list; float64
        List of true theta in order of sig, lam, gamma, rho, beta.
                
    Returns
    -------
    BLANK; prints table as side-effect.
     """
    t_sig, t_lam, t_gamma, t_rho = t_theta[:4]
    t_beta = t_theta[4:]
    k = len(t_beta)
    
    gam_ls = []
    rho_ls = []
    beta_ls = []
    lam_ls = []
    sig_ls = []
    
    n = len(sample_ls)
    for i in range(n):
        sig_ls.append(sample_ls[i][0])
        lam_ls.append(sample_ls[i][1])
        gam_ls.append(sample_ls[i][2])
        rho_ls.append(sample_ls[i][3])
        beta_ls.append(sample_ls[i][4])
        
    # bias
    gam_bias = np.array(gam_ls).mean()-t_gamma
    rho_bias = np.array(rho_ls).mean()-t_rho
    beta_map = map(np.mean, zip(*beta_ls))
    beta_bias = list(beta_map)
    for i in range(k):
        beta_bias[i] = beta_bias[i]-t_beta[i]
    lam_bias = np.array(lam_ls).mean()-t_lam
    sig_bias = np.array(sig_ls).mean()-t_sig
    bias = [sig_bias, lam_bias, gam_bias,rho_bias, beta_bias]
    
    # rmse
    gam_rmse = sqrt(mean_squared_error(np.array([t_gamma]*n), np.array(gam_ls)))
    rho_rmse = sqrt(mean_squared_error(np.array([t_rho]*n), np.array(rho_ls)))
    t_beta_list = np.repeat(t_beta,n).reshape(k,n)
    emp_beta = list(zip(*beta_ls))
    beta_rmse = []
    for i in range(k):
        beta_rmse.append(mean_squared_error(t_beta_list[i],emp_beta[i]))
    lam_rmse = sqrt(mean_squared_error(np.array([t_lam]*n), np.array(lam_ls)))
    sig_rmse = sqrt(mean_squared_error(np.array([t_sig]*n), np.array(sig_ls)))
    rmse = [sig_rmse, lam_rmse, gam_rmse,rho_rmse, beta_rmse]
    
    # sd
    gam_sd = np.array(gam_ls).std()
    rho_sd = np.array(rho_ls).std()
    lam_sd = np.array(lam_ls).std()
    sig_sd = np.array(sig_ls).std()
    temp = map(np.std, zip(*beta_ls))
    beta_sd = list(temp)
    sd = [sig_sd, lam_sd, gam_sd,rho_sd, beta_sd]
    
    # print table
    bias_ls = ["Bias", sig_bias, lam_bias, gam_bias,rho_bias, beta_bias]
    sd_ls = ["SD", sig_sd, lam_sd, gam_sd,rho_sd, beta_sd]
    rmse_ls = ["RMSE", sig_rmse, lam_rmse, gam_rmse,rho_rmse, beta_rmse]
    data = [bias_ls, sd_ls, rmse_ls]
    #define header names
    col_names = ["Measures (n={}; T={})".format(n, T), "sigma", "lambda", "gamma", "rho", "beta"]
    # display table
    print(tabulate(data, headers=col_names))
    
    
    
def generate_multi_table(n_ls,T_ls,t_theta_ls,g_theta_ls,W_ls, N=50, auto_weight = True, alternate = True, constrain=True):
    r"""generate multiple stats table for estimator peformance (each table is an output of a monte carlo experiement of N samples)
     
    The main idea of this function is to generate a series of table of stats for the performance of estimators from a series of monte-carlo experiments.
    
    Parameters
    ----------
    n_ls: list; int
        A list of total number of items at each time point.
    T_ls : list; int
        A list of totall number of time points.
    t_theta_ls : list; list; float64
        List containing list of true theta in order of sig, lam, gamma, rho, beta.
    g_theta_ls : list; list; float64
        List containing list of guessed theta in order of sig, lam, gamma, rho, beta.
    W_ls : list; list; ndarray; float64
        List of list of weight matrices, fixed for each monte-carlo experiment.
    N : int
        Number of monte carlo experiments for each table
    auto_weight : bool
        True if requires auto-generate weight-matrices for experiments.
    alternate : bool
        True if chooses alternating weight-matrices at odd/even time points
    constrain : bool
        True if the chosen optimization is constrained.
                        
    Returns
    -------
    total_W_ls : list; list; ndarray; float64
        If chosen auto, then returns generate weight matrix list. 
    
    prints tables as side-effect.
     """    
    num = len(n_ls)
    total_W_ls = []
    for i in range(num):
        if auto_weight:
            W_ls = generate_wls(n_ls[i],T_ls[i], alternate = alternate)
            total_W_ls.append(W_ls)
        sample_ls = generate_samples_scipy_fix_weight(n=n_ls[i],T=T_ls[i],t_theta=t_theta_ls[i],g_theta=g_theta_ls[i], N=N, W_ls = W_ls, constrain=constrain)
        # print tables
        obtain_table_stats(sample_ls,t_theta_ls[i])  
    if auto_weight:
        return total_W_ls