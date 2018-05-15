"""Input and output helpers to load in data.
(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    images = []
    labels = []
    with open(data_txt_file,'r') as raw_data:
        for line in raw_data:
            image,label = line.split(',')
            image_value = io.imread(image_data_path+image+'.jpg')
            images.append(image_value)
            labels.append(int(label.strip('\n')))
    data = {}
    data['image'] = np.array(images).astype(float)
    data['label'] = np.array(labels)
    return data
