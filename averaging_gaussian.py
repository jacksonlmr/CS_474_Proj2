from helpers import traverseImage, weightSumMatrix
import numpy as np
import cv2

outfile_save_path = "Output_Images/"

lenna = cv2.imread('Input_Images/lenna.gif', flags=0)
sf = cv2.imread('Input_Images/sf.gif', flags=0)

# print(lenna)
def average(input_img_array: np.ndarray, size: int):
    """
    Performs averaging on an image to smooth

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image. Should have dtype=np.uint8.

    >**size**:
    >int representing the size of the averaging mask

    **Returns**
    -----------
    >**output_array**: 2D array representing the averaged image.
    """
    #create normalized averaging matrix
    weights = np.ones((size, size))
    total_weights = weights.shape[0]*weights.shape[1]
    weights = weights * total_weights
    
    return traverseImage(input_img_array, weights, weightSumMatrix)

# lenna_average7 = average(lenna, 7)
# cv2.imwrite(f'{outfile_save_path}lenna_averaged7.jpg', lenna_average7)

# lenna_average15 = average(lenna, 15)
# cv2.imwrite(f'{outfile_save_path}lenna_averaged15.jpg', lenna_average15)

# average_test_array = np.array([
#     [0, 1, 2],
#     [2, 1, 0],
#     [0, 1, 2]
# ], dtype=np.uint8)

# print(average(average_test_array, 3))

def gaussian(input_img_array: np.ndarray, std_dev: int):
    size = 5*std_dev
    weights = np.random.normal(loc=0, scale=std_dev, size=(size, size))
    return traverseImage(input_img_array, weights, weightSumMatrix)

print(np.random.normal(loc=0, scale=1.4, size=7))