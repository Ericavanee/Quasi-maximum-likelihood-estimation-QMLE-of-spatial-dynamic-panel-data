import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from tabulate import tabulate

import pylab
import scipy.stats as stats

from var_and_coeff import *


# Authors: Authors: Agostino Capponi, Mohammadreza Bolandnazar, Erica Zhang
# License: MIT License
# Version: Nov 2nd, 2022

# DESCRIPTION: This package computes QMLE estimators with k-D features and exogenous, potentially time-varying weight matrices using scipy.optimize. Implementation is based on the QMLE model developed by Lee & Yu (2011): https://www.sciencedirect.com/science/article/pii/S0304407616302147


def QMLE_scipy_obj(params):
    r"""QMLE objective function formulated for scipy.optimize
     
    The main idea of this function is to generate the objective QMLE function formulated in equation (7) of the paper by Lee & Yu. 
    The objective function resolves to a real value given parameters. 
    
    Parameters
    ----------
    params: 1D array
        1D numpy array of sigma, lambda, gamma, rho, and beta, where beta is an 1D array of length k.
        
    Returns
    -------
    float64
        negative of QMLE real-valued output
     """ 
    sigma, lam, gamma, rho = params[:4]
    beta = params[4:]
    obj = -1/2*((n-1)*T)*np.log(2*np.pi)-1/2*((n-1)*T)*np.log(sigma**2)-T*np.log(1-lam)+np.sum(get_S(sigma,lam,gamma, rho, beta))-1/(2*sigma**2)*np.sum(get_V(sigma,lam,gamma, rho, beta))
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
    beta : 1Darray; float64
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
    beta : 1Darray; float64
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
        # already in shape (n,1)
        SY_sum = sum(SY_ls)
        # first component
        SY_tilde = np.matmul(S_nt,np.array(y[i+1]).reshape(n,1))-(1/T)*SY_sum        

        # second component
        Y_ls = []
        for q in range(T):
            Y_ls.append(y[q])
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
            # convert x to array
            X_ls.append(np.array(x[m]))
        X_sum = sum(X_ls)
        X_tilde = np.array(np.array(x[i])-(1/T)*X_sum).reshape(n,k)
              
        Z_tilde = [Y_tilde_lag,WY_tilde_lag,X_tilde]
        
        # third component
        l_n = np.ones(n).reshape(n,1)
        # forth component
        J_n = np.identity(n)-(1/n)*np.matmul(l_n,l_n.transpose())
        # fifth component
        Z_delta = (gamma*Z_tilde[0]).reshape(n,1)+(rho*Z_tilde[1]).reshape(n,1)+np.matmul(Z_tilde[2],np.array(beta).reshape(k,1)).reshape(n,1)
        V_tilde = np.array(SY_tilde-Z_delta).reshape(n,1)
        # putting everything together
        VJ = np.matmul(V_tilde.transpose(),J_n)
        VJV = np.matmul(VJ, V_tilde)
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


# create the result class for QMLE_scipy
class scipy_res:
    def __init__(self, parameters, c0, alpha, residual, asymptotic_var):
        sigma, lam, gamma, rho = parameters[:4]
        beta = parameters[4:]
        self.params = [sigma, lam, gamma, rho, beta]
        self.parameters = parameters
        self.c0 = c0
        self.alpha = alpha
        self.residual = residual
        self.asymptotic_var = asymptotic_var
    
    def residual_mean(self):
        return np.array(self.residual).mean()
    
    def residual_std(self):
        return np.array(self.residual).std()
    
    def plot_residual(self,t = 1, plot_all = False):
        resi = self.residual
        if plot_all == False:
            print("Plotting qqplot of residuals at time t =", t, ".")
            print()
            tp = t-1
            measurements = resi[tp].ravel()  
            stats.probplot(measurements, dist="norm", plot=pylab)
            pylab.show()
        else :
            T = len(resi)
            print("Plotting qqplot of residuals from t = 1 to t = ",T, "." )
            print()
            for i in range(T):
                measurements = resi[i].ravel()  
                stats.probplot(measurements, dist="norm", plot=pylab)
                pylab.show()
                
    
    def sigma(self):
        return self.parameters[0]
    
    def lam(self):
        return self.parameters[1]
    
    def gamma(self):
        return self.parameters[2]
    
    def rho(self):
        return self.parameters[3]
    
    def beta(self):
        return self.parameters[4:]
        
    def print_table(self):
        # display table of residuals
        print("Residuals:")
        residual_data = np.percentile(np.array(self.residual), [0, 25, 50, 75, 100]).reshape(1,5)
        residual_col_names = ["Min", "1Q", "Median", "3Q", "Max"]
        print(tabulate(residual_data, headers=residual_col_names,tablefmt="simple"))
        print()
        
        # display estimated parameters
        sigma, lam, gamma, rho = self.parameters[:4]
        beta = self.parameters[4:]
        print("Estimated Parameters:")
        param_data = np.array([sigma,lam,gamma,rho,beta],dtype=object).reshape(1,5)
        param_col_names = ["Sigma", "Lambda", "Gamma", "Rho", "Beta"]
        print(tabulate(param_data, headers=param_col_names,tablefmt="simple"))
        print()
        
        # display estimated alpha
        print("Estimated alpha (display first 10 rows): ")
        if len(self.alpha) > 10:
            alpha_data = np.array(self.alpha.tolist()[:10]).reshape(1,10)
        else:
            alpha_data = np.array(self.alpha).reshape(1,len(self.alpha))
        alpha_col_names = []
        for i in range(len(self.alpha)):
            alpha_col_names.append("alpha_"+str(i))
        print(tabulate(alpha_data, headers=alpha_col_names,tablefmt="simple"))
        print()
        
        # display estimated c0
        print("Estimated c0 (display first 10 rows): ")
        if len(self.c0) > 10:
            c0_data = np.array(self.c0.tolist()[:10]).reshape(1,10)
        else:
            c0_data = np.array(self.c0).reshape(1,len(self.c0))
        c0_col_names = []
        for i in range(len(self.c0)):
            c0_col_names.append("c0_"+str(i))
        print(tabulate(c0_data, headers=c0_col_names,tablefmt="simple"))
        print()
        
        # display asymptotic variance
        print("Estimated asympotitic variance: ")
        beta_len = len(beta)
        beta_names = []
        for i in range(beta_len):
            beta_names.append("Beta_"+str(i))
        asymp_var_names = ["Gamma","Rho"]
        asymp_var_names.extend(beta_names)
        asymp_var_names.extend(["Lambda", "Sigma^2"])
        asymp_var_data = []
        for i in range(beta_len+4):
            temp = [asymp_var_names[i]]
            temp.extend(self.asymptotic_var[i])
            asymp_var_data.append(temp)
        print(tabulate(asymp_var_data, headers=asymp_var_names,tablefmt="simple"))
        print()


def QMLE_scipy_estimate(x_dataset, y_attribute, Weight_ls, initial_guess = None, constrain = True):
    r"""Runs scipy.optimize for QMLE
     
    The main idea of this function is to generate QMLE estimators by solving the optimizing function in scipy. 
    
    Parameters
    ----------
    alpha_0 : 1D array; float64
        1D array of length T for vector of time effect.
    c_0: 1Darray; float64
        1D array of length n.
    x_dataset : list; ndarray; float64
        List of length T of ndarray of shape (n,k) for feature vector.
    y_attribute : list; 1D array; float64
        List of length T+1 of 1D array of length n for spatial unit feature vector.
    Weight_ls : list; np.ndarray; float64
        List of length T+1 of np.ndarray for weight matrix per time point.
    initial_guess : list; float64
        List of length 5 containing initial guess of each of the parameters; if None, function will set initial_guess to default.
    est_coeff: Bool
        True if requested to show estimated coefficients.
    constrain: Bool
        Boolean choosing whether or not to impose constraints on the QMLE problem.
              
    Returns
    -------
    object
        Returns a scipy.res object that contains attributes including parameters, c0, alpha, residual, and asymptotic_var.        
    """ 
    n_0 = x_dataset[0].shape[0]
    T_0 = len(x_dataset)
    k_0 = x_dataset[0].shape[1]
    if initial_guess == None:
        # set up default initial guess
        initial_guess = [1, 0.5, 1, 1, *np.ones(k_0)]
    global n
    n = n_0
    global T
    T = T_0
    global k 
    k = k_0
    global x
    x = x_dataset
    global y
    y = y_attribute
    global W_ls
    W_ls = Weight_ls
    # implement the constrained version
    if constrain == True:
        const = scipy_constraint(T)
        while True:
            try:
                res = minimize(QMLE_scipy_obj, initial_guess, method='trust-constr', constraints = const)
            except:
                 continue
            else:
                #the rest of the code
                break
        # return maximizing parameters 
        my_params = res.x
        sigma, lam, gamma, rho = my_params[:4]
        beta = my_params[4:]
        residual = get_residual(n,k,T,x,y,W_ls,my_params)
        c0 = get_c(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
        alpha = get_alpha(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
        asymp_var = get_asymp_var(my_params, x,y, n, T, k, W_ls)
        # now create the scipy_res object
        my_res = scipy_res(my_params,c0,alpha,residual,asymp_var)
        return my_res
    else:
        while True:
            try:
                res = minimize(QMLE_scipy_obj, initial_guess, method='nelder-mead', options={'xatol': 0.00001, 'disp': False})
            except:
                 continue
            else:
                #the rest of the code
                break
        my_params = res.x
        sigma, lam, gamma, rho = my_params[:4]
        beta = my_params[4:]
        residual = get_residual(n,k,T,x,y,W_ls,my_params)
        c0 = get_c(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
        alpha = get_alpha(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
        asymp_var = get_asymp_var(my_params, x,y, n, T, k, W_ls)
        # now create the scipy_res object
        my_res = scipy_res(my_params,c0,alpha,residual,asymp_var)
        return my_res
    
