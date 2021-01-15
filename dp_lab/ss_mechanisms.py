import numpy as np 
''' 
    Implementation of instance-specific additive noise mechanism as proposed by Nissim et al.

    Paper: Smooth Sensitivity and Sampling in Private Data Analysis 
        (https://cs-people.bu.edu/ads22/pubs/NRS07/NRS07-full-draft-v1.pdf)
'''
def cauchy_DP_noise_ss(data: np.ndarray, epsilon, query, ss, size):
    ''' 
        Smooth Sensitivity mechanism to achive epsilon-DP using 
        the Cauchy distribution as admisible noise distribution
    
    Args:
        data (ndarray): array to apply the query
        epsilon (float): privacy budget
        query (function): query to apply
        ss (function): beta-smooth upper bound function
        size (int): number of query calls.
    Returns:
        query result with added noise
         
    '''
    beta = epsilon/4
    alpha = epsilon/4
    smooth_sensitivity = ss(data, beta)
    print(smooth_sensitivity)
    cauchy_noise = np.array([np.tan(np.pi*(np.random.rand()-0.5)) for i in range(size)])
    return query(data) + smooth_sensitivity/alpha*cauchy_noise 

    
def laplace_DP_noise_ss(data: np.ndarray, epsilon, delta, query, ss, size):
    ''' 
        Smooth Sensitivity mechanism to achive epsilon-DP using 
        the Laplace distribution as admisible noise distribution
    
    Args:
        data (ndarray): array to apply the query
        epsilon (float): privacy budget
        query (function): query to apply
        ss (function): beta-smooth upper bound function
        size (int): number of query calls.
    Returns:
        query result with added noise
         
    '''
    beta = epsilon/(2*np.log(2/delta))
    alpha = epsilon/2
    unif = np.random.rand(size)
    smooth_sensitivity = ss(data, beta)
    print(smooth_sensitivity)
    laplace_noise = np.array([np.sign(unif[i]-0.5)*np.log(1-2*np.abs(unif[i]-0.5)) for i in range(size)]) 
    return query(data) + smooth_sensitivity/alpha*laplace_noise 

def gaussian_DP_noise_ss(data: np.ndarray, epsilon, delta, query, ss, size):
    ''' 
        Smooth Sensitivity mechanism to achive epsilon-DP using 
        the Gaussian distribution as admisible noise distribution
    
    Args:
        data (ndarray): array to apply the query
        epsilon (float): privacy budget
        query (function): query to apply
        ss (function): beta-smooth upper bound function
        size (int): number of query calls.
    Returns:
        query result with added noise
         
    '''
    beta = epsilon/(4*(1+np.log(2/delta)))
    alpha = epsilon/(5*(np.sqrt(2*np.log(2/delta))))
    smooth_sensitivity = ss(data, beta)
    normal_noise = np.array([np.sqrt(-2*np.log(np.random.rand())*np.cos(2*np.pi*np.random.rand())) for i in range(size)])
    return query(data) + smooth_sensitivity/alpha*normal_noise 