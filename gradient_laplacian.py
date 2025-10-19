import numpy as np
import cv2
from helpers import traverseImage, weightSumMatrix

outfile_save_path = "Output_Images/"

lenna = cv2.imread('Input_Images/lenna.gif', flags=0)
sf = cv2.imread('Input_Images/sf.gif', flags=0)

def gradient(input_img_array: np.ndarray, weights: np.ndarray):
    """
    Computes the gradient of an image with a given mask.

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image. Should have dtype=np.uint8.

    >**weights**:
    >np.ndarray representing the mask to be used. Should have dtype=np.uint8.

    **Returns**
    -----------
    >**output_array**: 2D array representing the correlated image.
    """
    return traverseImage(input_img_array, weights, weightSumMatrix)

prewitt_x = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

prewitt_y = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

lenna_prewitt_x = gradient(lenna, prewitt_x)
cv2.imwrite(f'{outfile_save_path}lenna_prewitt_x.jpg', lenna_prewitt_x)