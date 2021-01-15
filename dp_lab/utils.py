import numpy as np 

def binary_search(func: callable, delta, start, end, tol=1.5E-5, max_iter=500):
    iteration = 0
    mid = (start+end)/2
    while (np.abs(func(mid)-delta)<tol and iteration<max_iter):
        if func(mid) > delta:
            end = mid
        else:
            start = mid
        mid = (start+end)/2
        iteration +=1
    return mid

def median_index(x: np.ndarray):
    if (x.size%2):
        return int((x.size+1)/2)
    return int(x.size/2)

def index_search(r, x):
    L = 0
    R = x.size-1
    m = int((L+R)/2)
    while (R-L>1):
        if (r < x[m]):
            R = m
        else:
            L = m
        m = int((L+R)/2)
    return L + 1

def close_index(r, x):
    m = median_index(x)
    if x[m-1]<=r:
        return list(x[m-1:]).index(r)+m
    if x[m-1]>=r:
        return m-list(-np.sort(-x[:m])).index(r)
    
def median_utility(r, x: np.ndarray):
    n = x.size
    m = median_index(x)
    if r in x:
        i = close_index(r, x)
        return -np.abs(2*np.abs(m-i) - (n%2)*int(r<x[m-1]) - ((n+1)%2)*int(r>x[m-1]))
    else:
        m_1 = m + (n+1)%2
        i = index_search(r, x) + 1
        return -np.abs(2*np.abs(m_1-i) - ((n+1)%2)*int(r<x[m-1]) - ((n)%2)*int(r>x[m-1]))