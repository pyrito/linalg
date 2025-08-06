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
	
print(v)


kman_pca()

