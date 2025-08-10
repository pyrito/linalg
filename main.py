import numpy as np


# test vector, define globally
v = np.array([[123, 23, 34],[41,51.2,64],[7,8,9]])
k = 2

# my attempt at understanding pca by myself
def kman_pca():
    # center the matrix
    c_mean = np.mean(v, axis=0)
    center_v = v - c_mean
    cov_v = np.cov(center_v)
    # get the eigenvectors and eigenvalues
    e_val, e_v = np.linalg.eig(cov_v)
    print(e_val)
    print(e_v)

    # take the top k eigenvectors
    idx = np.argsort(np.abs(e_val))[::-1][0:k]
    
    topk_e_v = e_v[:, idx]
    print(topk_e_v)

    x_proj = np.matmul(center_v,topk_e_v)
    print(x_proj)

print(v)

def lu_factor():
    # Simple poor man's lu factorization
    n = v.shape[0]
    U = v.copy()
    L = np.eye(n, dtype=np.double)
    # must be square
    assert v.shape[0] == v.shape[1]
    for i in range(n-1):
        pivot = U[i][i]
        pivot_row = U[i,:]
        for j in range(i+1, n):
            factor = U[j][i] / pivot
            U[j,i:] = (U[j,i:] - (factor*pivot_row[i:]))
            L[j][i] = factor
    
    return L,U

def qr_factor(m):
    n = m.shape[0]
    Q = np.zeros((n,n))
    R = np.zeros((n,n))

    # iterate over each column
    for i in range(n):
        a = m[:,i]
        u = a
        for j in range(i):
            # project a onto q
            q = Q[:,j]
            R[j,i] = np.dot(q,a)
            u -= (np.dot(q,a))*q

        R[i,i] = np.linalg.norm(u)
        Q[:,i] = u / R[i,i] 
    return Q,R

A = np.array([[1,1,0],[1,0,1],[0,1,1]], dtype='float64')
Q,R = qr_factor(A)

print(Q)
print(R)
#kman_pca()

