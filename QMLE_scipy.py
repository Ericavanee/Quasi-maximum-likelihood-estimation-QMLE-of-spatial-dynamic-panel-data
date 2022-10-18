import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

# Authors: Authors: Agostino Capponi, Mohammadreza Bolandnazar, Erica Zhang
# License: MIT License
# Version: Oct 18, 2022

# DESCRIPTION: This package computes QMLE estimators with 1D feature and exogenous, potentially time-varying weight matrices using scipy.optimize. Implementation is based on the QMLE model developed by Lee & Yu (2011): https://www.sciencedirect.com/science/article/pii/S0304407616302147


def QMLE_obj_scipy(params):
    r"""QMLE objective function formulated for scipy.optimize
     
    The main idea of this function is to generate the objective QMLE function formulated in equation (7) of the paper by Lee & Yu. 
    The objective function resolves to a real value given parameters. 
    
    Parameters
    ----------
    params: 1D array
        1D numpy array of five real numbers: sigma, lambda, gamma, rho, and beta.
        
    Returns
    -------
    float64
        negative of QMLE real-valued output
     """ 
    sigma, lam, gamma, rho, beta = params
    obj= -1/2*((n-1)*T)*np.log(2*np.pi)-1/2*((n-1)*T)*np.log(sigma**2)-T*np.log(1-lam)+np.sum(get_S(sigma,lam,gamma, rho, beta))-1/(2*sigma**2)*np.sum(get_V(sigma,lam,gamma, rho, beta))
    #maximize is to minimize the negative
    return (-1)*obj


def get_S(sigma,lam,gamma, rho, beta):
     r"""Assists QMLE objective function
     
    The main idea of this function is to provide import parts to the objective QMLE function formulated in equation (7) of the paper by Lee & Yu. This function resolves to a real value given parameters. 
    
    Parameters
    ----------
    sigma : float64
        standard deviation of column vector V_nt in equation (1).
    lam : float64
        regression function parameter in equation (1).
    gamma : float64
        regression function parameter in equation (1)
    rho : float64
        regression function parameter in equation (1)
    beta : float64
        regression function parameter in equation (1)
        
    Returns
    -------
    list; float64
        A list of real numbers
    """    
    
    S_ls = []
    for i in range(T):
        W = W_ls[i+1]
        S_nt = np.identity(n) - lam*W
        # further impose that the determinant is positive to allow logarithm to work
        det_S = np.linalg.det(S_nt)
        S_ls.append(np.log(abs(det_S)))
    return S_ls


def get_V(sigma,lam,gamma, rho, beta):
     r"""Assists QMLE objective function
     
    The main idea of this function is to provide import parts to the objective QMLE function formulated in equation (7) of the paper by Lee & Yu. This function resolves to a real value given parameters. 
    
    Parameters
    ----------
    sigma : float64
        standard deviation of column vector V_nt in equation (1).
    lam : float64
        regression function parameter in equation (1).
    gamma : float64
        regression function parameter in equation (1)
    rho : float64
        regression function parameter in equation (1)
    beta : float64
        regression function parameter in equation (1)
        
    Returns
    -------
    list; float64
        A list of real numbers
    """ 
    V_ls = []
    for i in range(T):
        W = W_ls[i+1]
        S_nt = np.identity(n) - lam*W
        SY_ls = []
        for j in range(T):
            W2 = W_ls[j+1]
            S_nt2 = np.identity(n) - lam*W2
            SY_ls.append(np.matmul(S_nt2,np.array(y[j+1]).reshape(n,1)))
        SY_sum = sum(SY_ls)
        # first component
        SY_tilde = (np.matmul(S_nt,np.array(y[j+1]).reshape(n,1))-(1/T)*SY_sum).reshape(n,1)
        # second component
        Y_ls = []
        for k in range(T):
            Y_ls.append(y[k])
        Y_tilde_lag = np.array(y[i]-(1/T)*sum(Y_ls)).reshape(n,1)
        WY_ls = []
        for l in range(T):
            W3 = W_ls[l]
            WY_ls.append(np.matmul(W3,np.array(y[l]).reshape(n,1)))
        WY_sum = sum(WY_ls)
        W_lag = W_ls[i]
        WY_tilde_lag = np.matmul(W_lag, np.array(y[i]).reshape(n,1))-(1/T)*(WY_sum)
        X_ls = []
        for m in range(T):
            X_ls.append(x[m])
        X_sum = sum(X_ls)
        X_tilde = np.array(x[i]-(1/T)*X_sum).reshape(n,1)
        Z_tilde = [Y_tilde_lag,WY_tilde_lag,X_tilde]
        # third component
        l_n = np.ones(n).reshape(n,1)
        alpha_ln = alpha[i]*l_n
        # forth component
        J_n = np.identity(n)-(1/n)*np.matmul(l_n,l_n.transpose())
        # fifth component
        V_tilde = np.array(SY_tilde-(gamma*Z_tilde[0]+rho*Z_tilde[1]+beta*Z_tilde[2])).reshape(n,1)
        # putting everything together
        VJ = np.matmul(V_tilde.transpose(),J_n)
        VJV = np.matmul(VJ,V_tilde)
        V_ls.append(VJV)     
    return V_ls


def scipy_constraint(T):
     r"""Generate constraints for QMLE
     
    The main idea of this function is to generate minor constraints for QMLE. It is important to note that in general QMLE does not require constraints. This function only offers an alternative to run 'trust-constr' method from scipy to ensure that minor constraints are satisfied so the optimizer does not encounter invalid values in its iterations. 
    
    Parameters
    ----------
    T: int
        Number of time points.
        
    Returns
    -------
    list; objects
        A list of NonlinearConstraint objects from scipy.optimize
    """ 
    con0 = lambda params: params[0]
    lc0 = NonlinearConstraint(con0, 0, np.inf)
    con1 = lambda params: -params[1]+1
    lc1 = NonlinearConstraint(con0, 0, np.inf)
    my_con = [lc0,lc1]
    for i in range(T):
        temp = lambda params: np.linalg.det(np.identity(n)-params[1]*W_ls[i+1])
        nlc = NonlinearConstraint(temp, 0, np.inf)
        my_con.append(nlc)
    return my_con


def QMLE_scipy_estimate(n_0,T_0, alpha_0, c0, x_dataset, y_attribute, Weight_ls, initial_guess, constrain = True):
    r"""Runs scipy.optimize for QMLE
     
    The main idea of this function is to generate QMLE estimators by solving the optimizing function in scipy. 
    
    Parameters
    ----------
    n_0: int
        Number of samples per time point.
    T_0 : int
        Number of time points.
    alpha_0 : 1D array; float64
        1D array of length T for vector of time effect.
    x_dataset : list; 1D array; float64
        List of length T of 1D array of length n for feature vector.
    y_attribute : list; 1D array; float64
        List of length T+1 of 1D array of length n for spatial unit feature vector.
    Weight_ls : list; np.ndarray; float64
        List of length T+1 of np.ndarray for weight matrix per time point.
    initial_guess : list; float64
        List of length 5 containing initial guess of each of the parameters.
    constrain: Bool
        Boolean choosing whether or not to impose constraints on the QMLE problem.
              
    Returns
    -------
    1D array; float64
        A list of found estimators in the sequence of sigma, lambda, gamma, rho, and beta.    
    """ 
    
    global n
    n = n_0
    global T
    T = T_0
    global x
    x = x_dataset
    global y
    y = y_attribute
    global c
    c = c0
    global alpha
    alpha = alpha_0
    global W_ls
    W_ls = Weight_ls
    # implement the constrained version
    if constrain == True:
        const = scipy_constraint(T)
        res = minimize(QMLE_obj_scipy, initial_guess, method='trust-constr', constraints = const)
        # return maximizing parameters 
        return res.x
    else:
        res = minimize(QMLE_obj_scipy, initial_guess, method='nelder-mead', options={'xatol': 0.000001, 'disp': False})
        return res.x
    
