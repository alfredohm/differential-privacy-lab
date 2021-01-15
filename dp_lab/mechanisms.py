import numpy as np
from utils import binary_search
from math import erf

def laplace_DP_noise(data: np.ndarray, query: callable, size: int, epsilon, query_sensitivity):
    '''  Laplace mechanism implementation to achive differential privacy.
    
    Args:
        data (ndarray): array to apply the query.
        query (callable): query to apply.
        size (int): number of query calls.
        epsilon (float): privacy budget.
        query_sensitivity (float): l_1 sensitivity of the query.
    Returns:
        query result with added noise. 
    '''
    scale = query_sensitivity/epsilon
    # Inverse transform sampling: 
    laplace_noise = np.array([scale*np.sign(np.random.rand()-0.5)*np.log(1-2*np.abs(np.random.rand()-0.5)) for i in range(size)])
    return query(data) - laplace_noise

def gaussian_DP_noise(data: np.ndarray, query: callable, size: int, epsilon, delta, query_sensitivity):
    ''' Gaussian mechanism to achive differential privacy.
    
    Args:
        data (ndarray): array to apply the query.
        query (callable): query to apply.
        size (int): number of query calls.
        epsilon (float): privacy budget.
        delta (float): privacy relaxation.
        query_sensitivity (float): l_2 sensitivity of the query.
    Returns:
        query result with added noise.
    '''
    variance = (2*np.log(1.25/delta)*query_sensitivity**2)/epsilon**2
    # Box–Muller transform:
    normal_noise = np.array([np.sqrt(variance*-2*np.log(np.random.rand()))*np.cos(2*np.pi*np.random.rand()) for i in range(size)])
    
    return query(data) + normal_noise


def analytic_gaussian_DP_noise(data: np.ndarray, query: callable, epsilon, delta, query_sensitivity):
    ''' Analytic Gaussian mechanism to achive differential privacy.
    
    Args:
        data (ndarray): array to apply the query.
        query (callable): query to apply.
        size (int): number of query calls.
        epsilon (float): privacy budget.
        delta (float): privacy relaxation.
        query_sensitivity (float): l_2 sensitivity of the query.
    Returns:
        query result with added noise.
    '''
    guassian_CDF = lambda x: (1+erf(x/np.sqrt(2)))/2
    b_plus = lambda v: guassian_CDF(np.sqrt(v*epsilon))-np.exp(epsilon)*guassian_CDF(-np.sqrt(epsilon*(v+2)))
    b_minus = lambda v: guassian_CDF(-np.sqrt(v*epsilon))-np.exp(epsilon)*guassian_CDF(-np.sqrt(epsilon*(v+2)))

    delta_0 = b_plus(0)
    start=0
    end=1

    if delta >= delta_0:
        while b_plus(float(end))<=delta:
            start=end
            end=2*end
        v_sup = binary_search(b_plus, delta, start, end)
        alpha = np.sqrt(1+v_sup/2)-np.sqrt(v_sup/2)
    else:
        while b_minus(float(end))>=delta:
            start=end
            end=2*end
        v_inf = binary_search(b_plus, delta, start, end)
        alpha = np.sqrt(1+v_inf/2)+np.sqrt(v_inf/2)
    
    scale = alpha*query_sensitivity/np.sqrt(2*epsilon)

    return query(data) + scale*np.sqrt(-2*np.log(np.random.rand()))*np.cos(2*np.pi*np.random.rand())

