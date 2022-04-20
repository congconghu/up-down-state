# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:58:10 2022

@author: Congcong
"""
import numpy as np

def identify_up_down(y, p_initial=0.5, p_transition=[[0.1, 0.9], [0.9, 0.1]], 
           J=2, mu=-2, alpha=3, beta=0.01):
    """
    get hidden state probability. The algorithm is described in:
    Le, N. D., Leroux, B. G., Puterman, M. L., & Albert, P. S. (1992). 
    Exact likelihood evaluation in a Markov mixture model for time series 
    of seizure counts. Biometrics, 317-323.

    Parameters
    ----------
    y : list of int
        Number of spikes at each time point.
    p_initial : float, optional
         Probability of state being 1 at the beginning. The default is 0.5.
    p_transition : list, optional
         Probability of state transision. 
         p_transition[i][j] is the probability of transition from state i to j.
         The default is [[0.1, 0.9], [0.9, 0.1]).
    J : int, optional
         Number of time point to consider in history. The default is 2.
    mu : float, optional
         Baseline firing rate. The default is -2.
    alpha : float, optional
         Firing rate increased by UP state. The default is 3.
    beta : TYPE, optional
         Firing rate influence of history period. The default is 0.01.

    Returns
    -------
    p_s_given_Y: vector
        probability pf observing s=1 with given spikes and HMM parameters
        Pr(sk=1|theta)
    mu : float
    alpha : float
    beta : float
    p_initial : float
    p_transition : list

    """
   
    
    n = len(y)
    y = np.round(np.array(y)/max(y)*100)
    nk = [sum(y[k-J:k-1]) if k > J else sum(y[:k]) for k in range(n)]
    nk = np.array(nk)
    
    L = [0] # log likelyhood
    e = 100 
    tol = 1e-5
    c = 0
    maxiter = 1000
    
    while abs(e) > tol and c < maxiter:
        
        (p_s_given_Y, p_s_joint, 
        mu_new, alpha_new, beta_new, 
        p_initial_new, p_transition_new) = EM(
            n, y, nk, mu, alpha, beta, p_initial, p_transition, J)
        
        s = p_s_given_Y > 0.5
        lambda_k = get_lambda_k(s, nk, mu, alpha, beta)
        L_new = get_log_likelihood(y, lambda_k, p_initial, p_transition, p_s_joint)
        L.append(L_new)
        e = L[-1]- L[-2]
        print('iteration {}: {}\n'.format(c, e))
        
        mu = mu_new
        alpha = alpha_new
        beta = beta_new 
        p_initial = p_initial_new
        p_transition = p_transition_new
        
        c += 1
                
    return (p_s_given_Y, mu, alpha, beta, p_initial, p_transition)

def EM(n, y, nk, mu, alpha, beta, p_initial, p_transition, J):
    """
    expectation maximization algorithm (EM).
    The E-step gives probability of hidden states using assumed HMM parameters
    {mu, alpha, beta, p_initial, p_transition, J}
    The M-step update HMM parameters using hidden state from the E-step

    Parameters
    ----------
    n : int
        number of time points.
    y : vector
        Number of spikes at each time point.
    nk : vector
        number of spikes in history period.
    mu : float
         Baseline firing rate.
    alpha : float
         Firing rate increased by UP state.
    beta : float
         Firing rate influence of history period.
    p_initial : float
         Probability of state being 1 at the beginning.
    p_transition : list
         Probability of state transision. 
    J : int
         Number of time point to consider in history

    Returns
    -------
    p_s_given_Y: vector
        probability pf observing s=1 with given spikes and HMM parameters
        Pr(sk=1|theta)
    p_s_joint : list
        p_s_joint[i][j] is the joint probability of s[k-1]=i and s[k]=j,
        for i,j = 0,1
    mu : float
    alpha : float
    beta : float
    p_initial : float
    p_transition : list
    """
    
    # Pr(yk|sk=0) and Pr(yk|sk=1)
    p_yk_s1 = get_p_yk(y, np.ones(n), nk, mu, alpha, beta)
    p_yk_s1 = np.reshape(p_yk_s1, (n, 1)) 
    p_yk_s0 = get_p_yk(y, np.zeros(n), nk, mu, alpha, beta)
    p_yk_s0 = np.reshape(p_yk_s0, (n, 1)) 
    p_yk_given_s = np.concatenate((p_yk_s0, p_yk_s1), axis=1)
        
    # E-step
    # aj[k] = Pr(y0:k, sk = j)
    # bj[k] = Pr(yk:K | y0:k, sk = j)
    a0, a1, ae0, ae1 = get_ak(p_yk_given_s, p_initial, p_transition, J, mu, alpha, beta)
    b0, b1, be0, be1 = get_bk(p_yk_given_s, p_initial, p_transition, J, mu, alpha, beta)
    
    # Pr(Y=y)
    p_Y_e = int(max(ae0[-1], ae1[-1]))
    p_Y = a0[-1]*10**(int(ae0[-1])-p_Y_e) + a1[-1]*10**(int(ae1[-1])-p_Y_e)
    
    # Pr(sk=1, Y)            
    p_s_Y = a1*b1
    p_s_Y_e = ae1+be1
    # Pr(sk=1 | Y)
    p_s_given_Y = p_s_Y/p_Y * np.float_power(10, p_s_Y_e - p_Y_e)                
    p_initial = p_s_given_Y[0]
    if p_initial > 1-1e-10: p_initial = 1 - 1e-10
    
    
    # M-step
    # Pr(s[k-1]=i, s[k]=j | Y)
    p_s_joint_00 = (p_transition[0][0]*p_yk_given_s[1:,0]*a0[:-1]*b0[1:] / p_Y
                        *  np.float_power(10, ae0[:-1] + be0[1:] - p_Y_e))
    p_s_joint_01 = (p_transition[0][1]*p_yk_given_s[1:,1]*a0[:-1]*b1[1:] / p_Y
                        *  np.float_power(10, ae0[:-1] + be1[1:] - p_Y_e))
    p_s_joint_10 = (p_transition[1][0]*p_yk_given_s[1:,0]*a1[:-1]*b0[1:] / p_Y
                        *  np.float_power(10, ae1[:-1] + be0[1:] - p_Y_e))
    p_s_joint_11 = (p_transition[1][1]*p_yk_given_s[1:,1]*a1[:-1]*b1[1:] / p_Y
                        *  np.float_power(10, ae1[:-1] + be1[1:] - p_Y_e))
    p_s_joint = [[p_s_joint_00, p_s_joint_01],[p_s_joint_10, p_s_joint_11]]
    
    # Pr(sk = 0 | Y) and Pr(sk = 1 | Y)
    p_s_0 = p_s_joint_00 + p_s_joint_10
    p_s_1 = p_s_joint_01 + p_s_joint_11
    
    p_transition[0][1] = sum(p_s_joint_01)/sum(p_s_0)
    p_transition[1][0] = sum(p_s_joint_10)/sum(p_s_1)
    p_transition[0][0] = 1 - p_transition[0][1]
    p_transition[1][1] = 1 - p_transition[1][0]
    
    mu, alpha, beta = Newton_raphe(y, nk, p_s_given_Y, mu, alpha, beta, J)
    
    return (p_s_given_Y, p_s_joint, mu, alpha, beta, p_initial, p_transition)

def get_log_likelihood(y, lambda_k, p_initial, p_transition, p_s_joint):
    """
    get log likelihood of p(s, y | theta)
    theta = {p_initial, p_transition, mu, alpha, beta}

    Parameters
    ----------
    y : vector
        Number of spikes at each time point.
    lambda_k : float
        time-varing rate lambda, determined by the brain state at the time
    p_initial : float
        Probability of state being 1 at the beginning.
    p_transition : list
        Probability of state transision. 
    p_s_joint : list
        p_s_joint[i][j] is the joint probability of s[k-1]=i and s[k]=j,
        for i,j = 0,1

    Returns
    -------
    float
        log likelihood of p(s, y | theta)

    """
    term1 = sum(y*np.log(lambda_k) - lambda_k)
    term2 = sum(  p_s_joint[0][0]*p_transition[0][0] 
                + p_s_joint[0][1]*p_transition[0][1]
                + p_s_joint[1][0]*p_transition[1][0]
                + p_s_joint[1][1]*p_transition[1][1])
    return term1 + term2 + p_initial
    

def Newton_raphe(y, nk, s, mu, alpha, beta, J):
    """
    get updated mu, alpha, and beta using Newton_raphe method to solve 
    the following functions:
        sum(sk*yk) = sum( sk * exp(mu + alpha*sk + beta*nk) )
        sum(nk*yk) = sum( nk * exp(mu + alpha*sk + beta*nk) )
        sum(yk) = sum( exp(mu + alpha*sk + beta*nk) )
    which is derived from setting the derivative of the log likelihood function
    in respect to mu, alpha and beta to 0, respectively

    Parameters
    ----------
    y : vector
        Number of spikes at each time point.
    nk : vector
        number of spikes in history period.
    s : vector
        probability of state being UP at each time point. Pr(sk = 1)
    mu : float
         Baseline firing rate.
    alpha : float
         Firing rate increased by UP state.
    beta : float
         Firing rate influence of history period.
    J : int
         Number of time point to consider in history.

    Returns
    -------
    tuple
        updated (mu, alpha, beta)

    """
    
    func0 = lambda x: get_lambda_k(s[J:], nk[J:], x[0], x[1], x[2]) - y[J:]
    func1 = lambda x: sum(func0(x))
    func2 = lambda x: sum(func0(x)*nk[J:])
    func3 = lambda x: sum(func0(x)*s[J:])
    
    jac_func0 = lambda x: get_lambda_k(s[J:], nk[J:], x[0], x[1], x[2])
    jac_func11 = lambda x: sum(jac_func0(x))
    jac_func12 = lambda x: sum(jac_func0(x) *s[J:])
    jac_func13 = lambda x: sum(jac_func0(x) *nk[J:])
        
    jac_func21 = lambda x: sum(jac_func0(x)         *nk[J:])
    jac_func22 = lambda x: sum(jac_func0(x) *s[J:]  *nk[J:])
    jac_func23 = lambda x: sum(jac_func0(x) *nk[J:] *nk[J:])
    
    jac_func31 = lambda x: sum(jac_func0(x)         *s[J:])
    jac_func32 = lambda x: sum(jac_func0(x) *s[J:]  *s[J:])
    jac_func33 = lambda x: sum(jac_func0(x) *nk[J:] *s[J:])

    c = 0
    e = 100
    tol = 1e-10
    maxiter = 1000

    x_0 = np.array([mu, alpha, beta])
    while np.any(abs(e) > tol) and c < maxiter:
        
        Jac = [[jac_func11(x_0), jac_func12(x_0), jac_func13(x_0)], 
             [jac_func21(x_0), jac_func22(x_0), jac_func23(x_0)],
             [jac_func31(x_0), jac_func32(x_0), jac_func33(x_0)]]
        Jac = np.array(Jac)
        
        f = np.transpose([func1(x_0), func2(x_0), func3(x_0)])
        x_new = x_0 - np.dot(np.linalg.inv(Jac), f)
        
        e = x_new - x_0
        x_0 = x_new
        c += 1    

    return tuple(x_0)


def get_ak(p_yk, p_initial, p_transition, J, mu, alpha, beta):
    """
    get a0k and a1k for each time point k
    
    To avoid underflow, numbers are saved with 10-based scientific notation
    e0 and e1 stores power for each number in a. 

    Parameters
    ----------
    p_yk : array
        p_yk[:,0] is the probability of observing yk spikes given state 0;
        p_yk[:,1] is the probability of observing yk spikes given state 1.
    p_initial : float
        Probability of state being 1 at the beginning.
    p_transition : list
        Probability of state transision. 
        p_transition[i][j] is the probability of transition from state i to j.
    J : int
        Number of time point to consider in history.
    mu : float
        Baseline firing rate.
    alpha : float
        Firing rate increased by UP state.
    beta : float
        Firing rate influence of history period.

    Returns
    -------
    a0 : vector
        Pr(Y(1,t)=y(1,t), st=0) for all t <= k
    a1 : vector
        Pr(Y(1,t)=y(1,t), st=1) for all t <= k
    e0 : vector
    e1 : vector
    
    """
        
    a0 = []
    e0 = []
    a1 = []
    e1 = []
    n = len(p_yk)
    for k in range(n):
        # Pr(yk | s = 1 or 0)
        p_yk_1 = p_yk[k, 1]
        p_yk_0 = p_yk[k, 0]
        if k == 0:
            
            a1_tmp = p_initial*p_yk_1
            e1.append(int(np.floor(np.log10(a1_tmp))))
            a1.append(a1_tmp*10**(-e1[-1]))
            
            a0_tmp = (1-p_initial)*p_yk_0
            e0.append(int(np.floor(np.log10(a0_tmp))))
            a0.append(a0_tmp*10**(-e0[-1]))
            
        else:
            
            a00 = a0[-1]*p_yk_0*p_transition[0][0]
            a00_e = e0[-1] + int(np.floor(np.log10(a00)))
            a00 = a00*10**(e0[-1] - a00_e)
            
            a10 = a1[-1]*p_yk_0*p_transition[1][0]
            a10_e = e1[-1] + int(np.floor(np.log10(a10)))
            a10 = a10*10**(e1[-1] - a10_e)
            
            a01 = a0[-1]*p_yk_1*p_transition[0][1]
            a01_e = e0[-1] + int(np.floor(np.log10(a01)))
            a01 = a01*10**(e0[-1] - a01_e)
            
            a11 = a1[-1]*p_yk_1*p_transition[1][1]
            a11_e = e1[-1] + int(np.floor(np.log10(a11)))
            a11 = a11*10**(e1[-1] - a11_e)
            
            
            a1_tmp_e = max(a01_e, a11_e)
            a1_tmp = a01*10**(a01_e-a1_tmp_e)+a11*10**(a11_e-a1_tmp_e)
            e1.append(a1_tmp_e )
            a1.append(a1_tmp )
            
            a0_tmp_e = max(a10_e, a00_e)
            a0_tmp = a00*10**(a00_e-a0_tmp_e)+a10*10**(a10_e-a0_tmp_e)
            e0.append(a0_tmp_e )
            a0.append(a0_tmp )
        
    a0 = np.array(a0)
    a1 = np.array(a1) 
    e0 = np.array(e0)
    e1 = np.array(e1)
        
    return (a0, a1, e0, e1)


def get_bk(p_yk, p_initial, p_transition, J, mu, alpha, beta):
    """
    get b0k and b1k for each time point k
    
    To avoid underflow, numbers are saved with 10-based scientific notation
    e0 and e1 stores power for each number in b. 

    Parameters
    ----------
    p_yk : array
        p_yk[:, 0] is the probability of observing yk spikes given state 0;
        p_yk[:, 1] is the probability of observing yk spikes given state 1.
    p_initial : float, optional
        Probability of state being 1 at the beginning. The default is 0.5.
    p_transition : list, optional
        Probability of state transision. 
        p_transition[i][j] is the probability of transition from state i to j.
        The default is [[0.1, 0.9], [0.9, 0.1]).
    J : int, optional
        Number of time point to consider in history. The default is 2.
    mu : float, optional
        Baseline firing rate. The default is -2.
    alpha : float, optional
        Firing rate increased by UP state. The default is 3.
    beta : TYPE, optional
        Firing rate influence of history period. The default is 0.01.

    Returns
    -------
    b0 : vector
        Pr(Y(t+1,n)=y(t+1,n) given st=0) for all t >= k
    b1 :  vector
        Pr(Y(t+1,n)=y(t+1,n) given st=1) for all t >= k
    e0 : vector
    e1 : vector
    
    """
    b0 = [1]
    b1 = [1]
    e0 = [0]
    e1 = [0]
    n = len(p_yk)
    
    for k in range(n-2, -1, -1):
            p_yk_1 = p_yk[k+1, 1]
            p_yk_0 = p_yk[k+1, 0]
            
            b00 = b0[0]*p_yk_0*p_transition[0][0]
            b00_e = e0[0] + int(np.floor(np.log10(b00)))
            b00 = b00 * 10**(e0[0]-b00_e )
            
            b01 = b1[0]*p_yk_1*p_transition[0][1]
            b01_e = e1[0] + int(np.floor(np.log10(b01)))
            b01 = b01 * 10**(e1[0]-b01_e )
            
            b10 = b0[0]*p_yk_0*p_transition[1][0]
            b10_e = e0[0] + int(np.floor(np.log10(b10)))
            b10 = b10 * 10**(e0[0]-b10_e )
            
            b11 = b1[0]*p_yk_1*p_transition[1][1]
            b11_e = e1[0] + int(np.floor(np.log10(b11)))
            b11 = b11 * 10**(e1[0]-b11_e )
            
            b1_tmp_e = max(b11_e, b10_e)
            b1_tmp = b11*10**(b11_e-b1_tmp_e) +  b10*10**(b10_e-b1_tmp_e)
            e1.insert(0, b1_tmp_e)
            b1.insert(0, b1_tmp)
            
            b0_tmp_e = max(b01_e, b00_e)
            b0_tmp = b01*10**(b01_e-b0_tmp_e) +  b00*10**(b00_e-b0_tmp_e)
            e0.insert(0, b0_tmp_e)
            b0.insert(0, b0_tmp)
    
    b0 = np.array(b0)
    b1 = np.array(b1) 
    e0 = np.array(e0)
    e1 = np.array(e1)
    
    return (b0, b1, e0, e1)


def get_p_yk(y, s, nk, mu, alpha, beta):
    """
    get the probability of observing yk spikes at time k given brain state s
    Pr(yk|s)

    Parameters
    ----------
    y : vector
        number of spikes at each time k.
    s : vector
        state of the brain, {0, 1} for DOWN and UP state respectively.
    nk : vector
        number of spikes in history period.
    mu : float
        baseline firing rate. 
    alpha : float
        firing rate increased by UP state.
    beta : float
        firing rate influence of history period.

    Returns
    -------
    float
        the probability of observing yk spikes at time k given brain state s

    """
    n = len(y)
    lambda_k = get_lambda_k(s, nk, mu=mu, alpha=alpha, beta=beta)
    fac = np.array([np.math.factorial(y[k]) for k in range(n)])
    return np.exp(-lambda_k) * lambda_k**y / fac
    

def get_lambda_k(s, nk, mu, alpha, beta):
    """
    get the time-varing rate lambda at time k

    Parameters
    ----------
    s : vector
        state of the brain, {0, 1} for DOWN and UP state respectively.
    nk : int
        number of spikes in history period.
    mu : float
        baseline firing rate.
    alpha : float
        firing rate increased by UP state.
    beta : float
        firing rate influence of history period.

    Returns
    -------
    float
        time-varing rate lambda, determined by the brain state at the time

    """
    return np.exp(mu + alpha*s + beta*nk)