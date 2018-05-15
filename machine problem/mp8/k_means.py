from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset

# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)
# Write your code here!
def avg(arr):
	return sum(arr)/len(arr)

def k_means(C):
	C_old = deepcopy(C)
	TOL = 10**(-3)
	df = pd.read_csv("./data/data/iris.data").as_matrix()[:,:4]
	d = len(df)
	e1 = 1
	e2 = 1
	e3 = 1
	while (e1>TOL or e2>TOL or e3>TOL): 
		D = [[],[],[]]
		for i in range(d):
			points = deepcopy(df[i]).astype(float)
			index = np.argmin(np.linalg.norm(C_old-points,axis=1))
			D[index].append(points)

		C_new = np.array([avg(D[0]),avg(D[1]),avg(D[2])])
		e = C_new-C_old
		C_old = deepcopy(C_new)
		e1 = np.linalg.norm(e[0])
		e2 = np.linalg.norm(e[1])
		e3 = np.linalg.norm(e[2])
		
	C_final = deepcopy(C_new)
	print("C_final is: ")
	print(C_final)
	return C_final








