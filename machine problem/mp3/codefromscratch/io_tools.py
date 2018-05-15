"""Input and output helpers to load in data.
"""
import numpy as np

def read_dataset(path_to_dataset_folder,index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1],
                                                     [1, x2],
                                                     [1, x3],
                                                     .......]
                                where xi is the 16-dimensional feature of each sample

        T(numpy.ndarray): class label vector T = [y1, y2, y3, ...]
                             where yi is +1/-1, the label of each sample
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    A = []
    T = []

    with open(path_to_dataset_folder+'/'+index_filename,'r') as raw_data:
    	for line_a in raw_data:
    		label,data = line_a.split(' ')
    		T.append(int(label))
    		feature_v = [1]
    		with open(path_to_dataset_folder+'/'+data.strip('\n'),'r') as feature_file:
    			for line_b in feature_file:
    				for t in line_b.split():
    				    try:
    				        feature_v.append(float(t))
    				    except ValueError:
    				        pass
    		A.append(feature_v)

    A = np.array(A)
    T = np.array(T)

    return A,T
