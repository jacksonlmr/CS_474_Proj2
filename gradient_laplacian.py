import numpy as np
import cv2
from helpers import traverseImage, weightSumMatrix, mapValues
import math

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

def gradient_magnitude(x_values: np.ndarray, y_values: np.ndarray):
    """
    Computes the gradient magnitude of an image given the x and y gradients.

    **Parameters**
    ---------------
    >**x_values**:
    >np.ndarray representing the x gradient of the image. Should have dtype=np.uint8.

    >**y_values**:
    >np.ndarray representing the y gradient of the image. Should have dtype=np.uint8.

    **Returns**
    -----------
    >**output_array**: 2D array representing the gradient magnitude of the image.
    """
    #x and y will have same number of rows and cols
    rows, cols = x_values.shape

    gradient_mag_array = np.zeros(shape=(rows, cols), dtype=np.uint64)
    for row in range(rows):
        for col in range(cols):
            x_value = x_values[row, col]
            y_value = y_values[row, col]

            gradient_mag_value = int(math.sqrt((x_value**2)+(y_value**2)))
            print(f'x_value: {x_value}\ny_value: {y_value}\nmag: {gradient_mag_value}\n\n')
            gradient_mag_array[row, col] = gradient_mag_value

    gradient_mag_array = mapValues(gradient_mag_array)
    return gradient_mag_array

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

lenna_prewitt_y = gradient(lenna, prewitt_y)
cv2.imwrite(f'{outfile_save_path}lenna_prewitt_y.jpg', lenna_prewitt_y)

lenna_prewitt_mag = gradient_magnitude(lenna_prewitt_x, lenna_prewitt_y)
cv2.imwrite(f'{outfile_save_path}lenna_prewitt_mag.jpg', lenna_prewitt_mag)
# sf_prewitt_x = gradient(sf, prewitt_x)
# cv2.imwrite(f'{outfile_save_path}sf_prewitt_x.jpg', sf_prewitt_x)

# sf_prewitt_y = gradient(sf, prewitt_y)
# cv2.imwrite(f'{outfile_save_path}sf_prewitt_y.jpg', sf_prewitt_y)

