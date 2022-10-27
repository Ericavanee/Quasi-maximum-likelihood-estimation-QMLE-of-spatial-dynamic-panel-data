import numpy as np

# Authors: Authors: Agostino Capponi, Mohammadreza Bolandnazar, Erica Zhang
# License: MIT License
# Version: Oct 23, 2022

# DESCRIPTION: This package computes QMLE asymptotic variance, 'alpha' coefficient and 'c0' coefficient. Implementation is based on the QMLE model developed by Lee & Yu (2011): https://www.sciencedirect.com/science/article/pii/S0304407616302147

# test passed
def get_asymp_var(estimated_params, x,y, n, T, k, W_ls):
    sigma, lam, gamma, rho = estimated_params[:4]
    beta = estimated_params[4:]
    info_mat = get_info_mat(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
    var_score = get_var_score(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
    inv_info_mat = np.linalg.inv(info_mat)
    info_score = np.matmul(inv_info_mat,var_score)
    asymp_var = 1/((n-1)*T)*np.matmul(info_score,inv_info_mat)
    return asymp_var

# test passed
def get_var_score(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T):
    info_mat = get_info_mat(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
    omega = get_omega(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
    return np.array(info_mat+omega).reshape(k+4,k+4)

# test passed
def get_omega(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T):
    omega_first_part = get_omega_first_part(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
    omega_second_part = get_omega_second_part(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
    omega = omega_first_part+omega_second_part
    return np.array(omega).reshape(k+4,k+4)

# test passed
def get_omega_first_part(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T):
    mu3 = Vnt_moments(n,k,T,x,y,W_ls,params,2)
    first_vec = get_first_omega_vec(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T,mu3)
    second_vec = get_second_omega_vec(x,y,W_ls,lam,sigma,gamma,rho,beta,k,n,T,mu3) # 1*(k+2)
    first_k_plus_2_row = []
    for i in range(k+2):
        my_row = np.zeros(k+2).tolist()
        my_row.append(first_vec[i])
        my_row.append(second_vec[i])
        first_k_plus_2_row.append(my_row)
        
    second_row = first_vec.tolist()
    second_row.extend([0,0])
    third_row = second_vec.tolist()
    third_row.extend([0,0])
    
    reshape_ls = first_k_plus_2_row
    reshape_ls.extend([second_row,third_row])
    print(reshape_ls)
    
    return np.array(reshape_ls).reshape(k+4,k+4)


# test passed
def get_first_omega_vec(x,y,W_ls,lam,sigma,gamma,rho,beta,k,n,T,mu3):
    sum_ls = []
    c0 = get_c(sigma,lam,gamma, rho, beta)
    for t in range(T):
        Gnt = get_Gnt(W_ls, n, lam, t)
        l_n = np.ones(n).reshape(n,1)
        J_n = np.identity(n)-(1/n)*np.matmul(l_n,l_n.transpose())
        JG = np.matmul(J_n,Gnt).reshape(n,n)
        JG_diag = JG.diagonal()
        Z_tilde = get_Z_tilde(n,k,x,y,W_ls,T,t)
        JZ_tilde = np.matmul(J_n,Z_tilde).reshape(n,k+2)
        row_vec = [] # 1*(k+2)
        for i in range(n):
            row_vec.append(JG_diag[i]*JZ_tilde[i])
        sum_ls.append(sum(row_vec))
    first_omega_vec = mu3/(sigma**4*(n-1)*T)*sum(sum_ls)
    return np.array(first_omega_vec)


# test passed
def get_second_omega_vec(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T,mu3):
    sum_ls = []
    for t in range(T):
        Z_tilde = get_Z_tilde(n,k,x,y,W_ls,T,t)
        l_n = np.ones(n).reshape(n,1)
        J_n = np.identity(n)-(1/n)*np.matmul(l_n,l_n.transpose())
        JZ_tilde = np.matmul(J_n,Z_tilde).reshape(n,k+2)
        row_vec = [] # 1*(k+2)
        for i in range(n):
            row_vec.append(JZ_tilde[i])
        sum_ls.append(sum(row_vec))
    second_omega_vec = mu3/(2*sigma**6*n*T)*sum(sum_ls)
    return np.array(second_omega_vec)


# test passed
def get_omega_second_part(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T):
    mu4 = Vnt_moments(n,k,T,x,y,W_ls,params,4)
    result = get_omega_second_part_first_and_second(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
    first_num = result[0]
    second_num = result[1]
    first_k_plus_2_row = []
    for i in range(k+2):
        first_k_plus_2_row.append(np.zeros(k+4))
    second_row = np.zeros(k+2).tolist()
    second_row.extend([first_num,second_num])
    third_row = np.zeros(k+2).tolist()
    third_row.extend([second_num,(mu4-3*sigma**4)/(4*sigma**8)])
    reshape_ls = first_k_plus_2_row
    reshape_ls.extend([second_row,third_row])
    
    return np.array(reshape_ls).reshape(k+4,k+4)


# test passed
def get_omega_second_part_first_and_second(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T):
    c0 = get_c(sigma,lam,gamma, rho, beta)
    mu3 = Vnt_moments(n,k,T,x,y,W_ls,params,3)
    mu4 = Vnt_moments(n,k,T,x,y,W_ls,params,4)
    first_sum_ls = []
    second_sum_ls = []
    third_sum_ls = []
    forth_sum_ls = []
    for t in range(T):
        delta = np.array([gamma,rho,*beta]).reshape(k+2,1)
        Gnt = get_Gnt(W_ls, n, lam, t)
        G_ls = []
        for m in range(T):
            G_ls.append(get_Gnt(W_ls, n, lam, m))
        G_tilde = Gnt-(1/T)*sum(G_ls)
        l_n = np.ones(n).reshape(n,1)
        J_n = np.identity(n)-(1/n)*np.matmul(l_n,l_n.transpose())
        Znt = get_Znt(n,k,x,y,W_ls,t)
        GZnt = np.matmul(Gnt,Znt).reshape(n,k+2)
        GZ_ls = []
        for r in range(T):
            Gnt2 = get_Gnt(W_ls, n, lam, r)
            Znt2 = get_Znt(n,k,x,y,W_ls,r)
            GZ_ls.append(np.matmul(Gnt2,Znt2))
        GZ_tilde = np.matmul(Gnt,Znt)-(1/T)*sum(GZ_ls)
        # arange components
        JG = np.matmul(J_n,Gnt).reshape(n,n)
        JG_sqr = np.matmul(JG,JG.transpose()).reshape(n,n)
        JG_diag = JG.diagonal()
        JG_sqr_diag = JG_sqr.diagonal()
        
        JGZ_tilde = np.matmul(J_n,GZ_tilde).reshape(n,k+2)
        JGZ_tilde_delta = np.matmul(JGZ_tilde,delta).reshape(n,1)
        
        JG_tilde = np.matmul(J_n,G_tilde).reshape(n,n)
        JG_tilde_c = np.matmul(JG_tilde,c0).reshape(n,1)
        
        JGZDelta_c = JGZ_tilde_delta+JG_tilde_c
        
        first_row_ls = []
        second_row_ls = []
        third_row_ls = []
        forth_row_ls = []
        
        for i in range(n):
            first_row_ls.append(*(JG_diag[i]*JGZDelta_c[i]))
            second_row_ls.append(JG_sqr_diag[i])
            third_row_ls.append(*(JGZDelta_c[i]))
            forth_row_ls.append(JG.trace())
        
        first_sum_ls.append(sum(first_row_ls))
        second_sum_ls.append(sum(second_row_ls))
    
    # first term
    first_term = (2*mu3)/(sigma**4*(n-1)*T)*sum(first_sum_ls)
    # second term
    second_term = (mu4-3*sigma**4)/(sigma**4*(n-1)*T)*sum(second_sum_ls)
    # third term
    third_term = (mu3)/(2*sigma**6*n*T)*sum(third_sum_ls)
    # forth term
    forth_term = (mu4-3*sigma**4)/(2*sigma**6*(n-1)*T)
    
    #altogether
    return [first_term+second_term,third_term+forth_term]


# test passed
def get_info_mat(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T):
    # this is a (k+4)*(k+4) matrix
    first_info_mat = get_first_infoMat(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
    second_info_mat = get_second_infoMat(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
    info_mat = 1/(sigma**2)*first_info_mat+second_info_mat
    info_mat.reshape(k+4,k+4)
    return info_mat


# test passed

def get_first_infoMat(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T):
    # (k+4)*(k+4)
    HnT = get_H_mat(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T)
    reshape_ls = []
    for i in range(k+3):
        row_vec = HnT[i].tolist()
        row_vec.append(0) # assume that * = 0
        reshape_ls.append(row_vec)
    reshape_ls.append(np.zeros(k+4))
    return np.array(reshape_ls).reshape(k+4,k+4)  


# test passed

def get_second_infoMat(x,y, W_ls,lam,sigma,gamma,rho,beta,k,n,T):
    # (k+4)*(k+4)
    GJG_JG_tr_ls = []
    JG_tr_ls = []
    for i in range(T):
        Gnt = get_Gnt(W_ls, n, lam, i)
        l_n = np.ones(n).reshape(n,1)
        J_n = np.identity(n)-(1/n)*np.matmul(l_n,l_n.transpose())
        GJ = np.matmul(Gnt.transpose(),J_n).reshape(n,n)
        GJG = np.matmul(GJ,Gnt).reshape(n,n)
        JG = np.matmul(J_n,Gnt).reshape(n,n)
        GJG_tr = np.matrix.trace(GJG)
        JG_tr = np.matrix.trace(JG)
        JG_sqr_tr = np.matrix.trace(np.matmul(JG,JG.transpose()))
        GJG_JG_tr = GJG_tr+JG_tr
        GJG_JG_tr_ls.append(GJG_JG_tr)
        JG_tr_ls.append(JG_tr)
    
    # constant
    first_trace = 1/((n-1)*T)*sum(GJG_JG_tr_ls)
    # constant
    second_trace = 1/(sigma**2*(n-1)*T)*sum(JG_tr_ls)
    
    # putting togther the matrix
    first_k_plus_2_row = []
    for j in range(k+2):
        first_k_plus_2_row.append(np.zeros(k+4))
    
    
    second_row = np.zeros(k+2).tolist()
    second_row.append(first_trace)
    second_row.append(second_trace)
    third_row = np.zeros(k+2).tolist()
    third_row.append(second_trace)
    third_row.append(1/(2*sigma**4))
    reshape_ls = first_k_plus_2_row
    reshape_ls.extend([second_row,third_row])
    
    return np.array(reshape_ls).reshape(k+4,k+4)        
     


# test passed

def get_H_mat(x,y,W_ls,lam,sigma,gamma,rho,beta,k,n,T):
    Hnt_vec_ls = []
    c0 = get_c(sigma,lam,gamma, rho, beta)
    for i in range(T):
        #delta is (k+2)*1
        delta = [gamma,rho,*beta]
        #Gnt is n*n
        Gnt = get_Gnt(W_ls, n, lam, i)
        #G_tilde is n*n
        G_ls = []
        for j in range(T):
            G_ls.append(get_Gnt(W_ls, n, lam, j))
        G_tilde = Gnt-(1/T)*sum(G_ls)
    
        #Znt and Z_tilde is n*(k+2)
        Znt = get_Znt(n,k,x,y,W_ls,i)
        Z_tilde = get_Z_tilde(n,k,x,y,W_ls,T,i)

        #GZnt and GZ_tilde is n*(k+2)
        GZnt = np.matmul(Gnt,Znt).reshape(n,k+2)
        GZ_ls = []
        for r in range(T):
            Gnt2 = get_Gnt(W_ls, n, lam, r)
            Znt2 = get_Znt(n,k,x,y,W_ls,r)
            GZ_ls.append(np.matmul(Gnt2,Znt2))
        GZ_tilde = np.matmul(Gnt,Znt)-(1/T)*sum(GZ_ls)
    
        #GZ_delta is n*1
        GZ_delta = np.matmul(GZ_tilde,delta).reshape(n,1)
        GZ_delta_c = GZ_delta - np.matmul(G_tilde, np.array(c0).reshape(n,1))
    
        # reshape H_mat
        ravel_Z_tilde = []
        for p in range(n):
            ravel_Z_tilde.append(Z_tilde[p])
    
        reshape_list = [ravel_Z_tilde,GZ_delta_c.ravel()]
        
        final_reshape_list = []
        for q in range(n):
            row_vec = reshape_list[0][q].tolist()
            row_vec.append(reshape_list[1][q])
            final_reshape_list.append(row_vec)
        
        H_mat = np.array(final_reshape_list).reshape(n,k+3)
        
        #J_n is n*n
        l_n = np.ones(n).reshape(n,1)
        J_n = np.identity(n)-(1/n)*np.matmul(l_n,l_n.transpose())
        
        #Hnt_vec is (k+3)*(k+3)
        Hnt_Jn = np.matmul(H_mat.transpose(),J_n)
        Hnt_vec = np.matmul(Hnt_Jn,H_mat)
        Hnt_vec_ls.append(Hnt_vec)
        
    H = (1/((n-1)*T))*sum(Hnt_vec_ls)
    return H        


# test passed
def get_Znt(n,k,x,y,W_ls,t):
    Y_lag = np.array(y[t]).reshape(n,1)
    WY_lag = np.matmul(W_ls[t],Y_lag).reshape(n,1)
    Xnt = np.array(x[t]).reshape(n,k)
    Znt = regroup_matrix(Y_lag,WY_lag,Xnt,n,k)
    return Znt


# test passed
def get_Gnt(W_ls, n, lam, t):
    W = W_ls[t]
    S_nt = np.identity(n) - lam*W
    Gnt = np.matmul(W,np.linalg.inv(S_nt)).reshape(n,n)
    return Gnt


# test passed
def get_Z_tilde(n,k,x,y,W_ls,T,t):
    Y_ls = []
    for q in range(T):
        Y_ls.append(y[q])
    Y_tilde_lag = np.array(y[t]-(1/T)*sum(Y_ls)) # flattened array
        
    WY_ls = []
    for l in range(T):
        W3 = W_ls[l]
        WY_ls.append(np.matmul(W3,np.array(y[l]).reshape(n,1)))
    WY_sum = sum(WY_ls)
    W_lag = W_ls[t]
    WY_tilde_lag = np.matmul(W_lag, np.array(y[t]).reshape(n,1))-(1/T)*(WY_sum)
    # flatten shape
    WY_tilde_lag.ravel()
        
    X_ls = []
    for m in range(T):
        X_ls.append(np.array(x[m]))
    X_sum = sum(X_ls)
    X_tilde = np.array(np.array(x[t])-(1/T)*X_sum).reshape(n,k) 
    
    ravel_X = []
    for p in range(n):
        ravel_X.append(X_tilde[p])
    
    #Z_tilde is n*(k+2)
    Z_tilde = regroup_matrix(Y_tilde_lag,WY_tilde_lag,ravel_X, n, k)
            
    return Z_tilde  


# test passed
def regroup_matrix(y_vec, wy_vec, x_vec, n, k):
    # regroup n*1, n*1, n*k into a matrix of n*(k+2)
    my_list = [y_vec,wy_vec,x_vec]
    reshape_list = []
    for i in range(n):
        row_vec = [*my_list[0][i],*my_list[1][i]]
        row_vec.extend(my_list[2][i])
        reshape_list.append(row_vec)
    return np.array(reshape_list).reshape(n,k+2)  


# get alpha and c

# test passed
def get_rnt(sigma,lam,gamma, rho, beta, t):
    # first part
    delta = np.array([gamma,rho,*beta]).reshape(k+2,1)
    W = W_ls[t+1]
    S_nt = np.identity(n) - lam*W # n-by-n
    SlamY = np.matmul(lam*S_nt,y[t+1]).reshape(n,1) # n-by-1
    # second part
    Znt = get_Znt(n,k,x,y,W_ls,t) # n-by-(k+2)
    Zdelta = np.matmul(Znt,delta).reshape(n,1)
    # mesh together
    rnt = SlamY-Zdelta
    return rnt


# test passed
def get_c(sigma,lam,gamma, rho, beta):
    sum_ls = []
    for i in range(T):
        rnt = get_rnt(sigma,lam,gamma, rho, beta, i)
        sum_ls.append(rnt)
    c0 = (1/T)*sum(sum_ls)
    return c0  


# test passed
def get_alpha_t(sigma,lam,gamma, rho, beta, t):
    l_n = np.ones(n).reshape(n,1)
    rnt = get_rnt(sigma,lam,gamma, rho, beta, t)
    c0 = get_c(sigma,lam,gamma, rho, beta)
    alpha_t = (1/n)*np.matmul(l_n.transpose(),rnt-c0)[0][0]
    return alpha_t

def get_alpha(sigma,lam,gamma, rho, beta, T):
    alpha = []
    for i in range(T):
        alpha_t = get_alpha_t(sigma,lam,gamma, rho, beta, i)
        alpha.append(alpha_t)
    return alpha

# test passed
def get_Vnt(n,k,T,x,y,W_ls,params,t):
    # get components
    sigma, lam, gamma, rho = params[:4]
    beta = params[4:]
    ynt = np.array(y[t+1]).reshape(n,1)
    y_lag = np.array(y[t]).reshape(n,1)
    W = np.array(W_ls[t+1]).reshape(n,n)
    W_lag = np.array(W_ls[t]).reshape(n,n)
    xnt = np.array(x[t]).reshape(n,k)
    c0 = get_c(sigma,lam,gamma, rho, beta)
    alpha_t = get_alpha_t(sigma,lam,gamma, rho, beta, t)
    l_n = np.ones(n).reshape(n,1)
    
    # get LHS
    LHS = lam*np.matmul(W,ynt)+gamma*y_lag+rho*np.matmul(W_lag,y_lag)+np.matmul(xnt,np.array(beta).reshape(k,1))+c0+alpha_t*l_n
    
    # get Vnt
    Vnt = np.array(ynt-LHS).reshape(n,1)
    
    return Vnt

# test passed
def get_residual(n,k,T,x,y,W_ls,params):
    Vnt_ls = []
    for i in range(T):
        Vnt = get_Vnt(n,k,T,x,y,W_ls,params,i)
        Vnt_ls.append(Vnt)
        
    return Vnt_ls


# test passed
def Vnt_moments(n,k,T,x,y,W_ls,params,moment):
    # moment is positive integer
    V_moment_ls = []
    for i in range(T):
        V_nt = get_Vnt(n,k,T,x,y,W_ls,params,i)
        V_moment = V_nt**moment
        V_moment_ls.append(V_moment)
    return np.array(V_moment_ls).mean()