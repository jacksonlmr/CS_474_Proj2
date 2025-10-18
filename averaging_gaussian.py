from helpers import traverseImage, weightSumMatrix
import numpy as np
import cv2

outfile_save_path = "Output_Images/"

lenna = cv2.imread('Input_Images/lenna.gif', flags=0)
sf = cv2.imread('Input_Images/sf.gif', flags=0)

cv2.imwrite(f'{outfile_save_path}lenna.jpg', lenna)

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
    weights = weights * (1/total_weights)
    
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

def gaussian(input_img_array: np.ndarray, weights: np.ndarray):
    """
    Computes the guassian of an image with a given mask.

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image. Should have dtype=np.uint8.

    >**weights**:
    >np.ndarray representing the mask to be used. Should have dtype=np.uint8.

    **Returns**
    -----------
    >**output_array**: 2D array representing the guassian blurred image.
    """
    #normalize weights
    weights = weights*(1/np.sum(weights))
    return traverseImage(input_img_array, weights, weightSumMatrix)

gaussian7 = np.array([
    [1, 1, 2, 2, 2, 1, 1],
    [1, 2, 2, 4, 2, 2, 1],
    [2, 2, 4, 8, 4, 2, 2],
    [2, 4, 8, 16, 8, 4, 2],
    [2, 2, 4, 8, 4, 2, 2],
    [1, 2, 2, 4, 2, 2, 1],
    [1, 1, 2, 2, 2, 1, 1]
])

gaussian15 = np.array([
    [2,2,3,4,5,5,6,6,6,5,5,4,3,2,2],
    [2,3,4,5,7,7,8,8,8,7,7,5,4,3,2],
    [3,4,6,7,9,10,10,11,10,10,9,7,6,4,3],
    [4,5,7,9,10,12,13,13,13,12,10,9,7,5,4],
    [5,7,9,11,13,14,15,16,15,14,13,11,9,7,5],
    [5,7,10,12,14,16,17,18,17,16,14,12,10,7,5],
    [6,8,10,13,15,17,19,19,19,17,15,13,10,8,6],
    [6,8,11,13,16,18,19,20,19,18,16,13,11,8,6],
    [6,8,10,13,15,17,19,19,19,17,15,13,10,8,6],
    [5,7,10,12,14,16,17,18,17,16,14,12,10,7,5],
    [5,7,9,11,13,14,15,16,15,14,13,11,9,7,5],
    [4,5,7,9,10,12,13,13,13,12,10,9,7,5,4],
    [3,4,6,7,9,10,10,11,10,10,9,7,6,4,3],
    [2,3,4,5,7,7,8,8,8,7,7,5,4,3,2],
    [2,2,3,4,5,5,6,6,6,5,5,4,3,2,2]
])

# lenna_guassian7 = gaussian(lenna, gaussian7)
# cv2.imwrite(f'{outfile_save_path}lenna_guassian7.jpg', lenna_guassian7)

# lenna_guassian15 = gaussian(lenna, gaussian15)
# cv2.imwrite(f'{outfile_save_path}lenna_guassian15.jpg', lenna_guassian15)