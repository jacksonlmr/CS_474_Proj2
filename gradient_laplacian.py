import numpy as np
import cv2
from helpers import traverseImage, weightSumMatrix, mapValues, add_images
import math

outfile_save_path = "Output_Images/Gradient_Laplacian/"

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

    x_values = x_values.astype(np.uint64)
    y_values = y_values.astype(np.uint64)
    #calculate the magnitude of the gradient at every pixel in the image
    gradient_mag_array = np.zeros(shape=(rows, cols), dtype=np.uint64)
    for row in range(rows):
        for col in range(cols):
            x_value = x_values[row, col]
            y_value = y_values[row, col]

            gradient_mag_value = math.sqrt((x_value**2)+(y_value**2))
            print(f'x_value: {x_value}\ny_value: {y_value}\nmag: {gradient_mag_value}\n\n')
            gradient_mag_array[row, col] = gradient_mag_value

    gradient_mag_array = mapValues(gradient_mag_array)
    return gradient_mag_array

#prewitt
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

# #lenna prewitt
# lenna_prewitt_x = gradient(lenna, prewitt_x)
# cv2.imwrite(f'{outfile_save_path}lenna_prewitt_x.jpg', lenna_prewitt_x)

# lenna_prewitt_y = gradient(lenna, prewitt_y)
# cv2.imwrite(f'{outfile_save_path}lenna_prewitt_y.jpg', lenna_prewitt_y)

# lenna_prewitt_mag = gradient_magnitude(lenna_prewitt_x, lenna_prewitt_y)
# cv2.imwrite(f'{outfile_save_path}lenna_prewitt_mag.jpg', lenna_prewitt_mag)

# #lenna prewitt sharpen
# lenna_prewitt_mag = lenna_prewitt_mag.astype(np.int64)
# prewitt_sharpened_lenna = add_images(lenna, -1*lenna_prewitt_mag)
# cv2.imwrite(f'{outfile_save_path}prewitt_sharpened_lenna.jpg', prewitt_sharpened_lenna)

# #sf prewitt
# sf_prewitt_x = gradient(sf, prewitt_x)
# cv2.imwrite(f'{outfile_save_path}sf_prewitt_x.jpg', sf_prewitt_x)

# sf_prewitt_y = gradient(sf, prewitt_y)
# cv2.imwrite(f'{outfile_save_path}sf_prewitt_y.jpg', sf_prewitt_y)

# sf_prewitt_mag = gradient_magnitude(sf_prewitt_x, sf_prewitt_y)
# cv2.imwrite(f'{outfile_save_path}sf_prewitt_mag.jpg', sf_prewitt_mag)

# #sf prewitt sharpen
# sf_prewitt_mag = sf_prewitt_mag.astype(np.int64)
# prewitt_sharpened_sf = add_images(sf, -1*sf_prewitt_mag)
# cv2.imwrite(f'{outfile_save_path}prewitt_sharpened_sf.jpg', prewitt_sharpened_sf)


#sobel
sobel_x = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

sobel_y = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# #lenna sobel
# lenna_sobel_x = gradient(lenna, sobel_x)
# cv2.imwrite(f'{outfile_save_path}lenna_sobel_x.jpg', lenna_sobel_x)

# lenna_sobel_y = gradient(lenna, sobel_y)
# cv2.imwrite(f'{outfile_save_path}lenna_sobel_y.jpg', lenna_sobel_y)

# lenna_sobel_mag = gradient_magnitude(lenna_sobel_x, lenna_sobel_y)
# cv2.imwrite(f'{outfile_save_path}lenna_sobel_mag.jpg', lenna_sobel_mag)

# #lenna sobel sharpen
# lenna_sobel_mag = lenna_sobel_mag.astype(np.int64)
# sobel_sharpened_lenna = add_images(lenna, -1*lenna_sobel_mag)
# cv2.imwrite(f'{outfile_save_path}sobel_sharpened_lenna.jpg', sobel_sharpened_lenna)

# #sf sobel
# sf_sobel_x = gradient(sf, sobel_x)
# cv2.imwrite(f'{outfile_save_path}sf_sobel_x.jpg', sf_sobel_x)

# sf_sobel_y = gradient(sf, sobel_y)
# cv2.imwrite(f'{outfile_save_path}sf_sobel_y.jpg', sf_sobel_y)

# sf_sobel_mag = gradient_magnitude(sf_sobel_x, sf_sobel_y)
# cv2.imwrite(f'{outfile_save_path}sf_sobel_mag.jpg', sf_sobel_mag)

# #sf sobel sharpen
# sf_sobel_mag = sf_sobel_mag.astype(np.int64)
# sobel_sharpened_sf = add_images(sf, -1*sf_sobel_mag)
# cv2.imwrite(f'{outfile_save_path}sobel_sharpened_sf.jpg', sobel_sharpened_sf)

laplacian_mask = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

lenna_laplacian = gradient(lenna, laplacian_mask)
cv2.imwrite(f'{outfile_save_path}lenna_laplacian.jpg', lenna_laplacian)

#allow negative values in array
lenna_laplacian = lenna_laplacian.astype(np.int64)
laplacian_sharpened_lenna = add_images(lenna, -1*lenna_laplacian)
cv2.imwrite(f'{outfile_save_path}laplacian_sharpened_lenna.jpg', laplacian_sharpened_lenna)


sf_laplacian = gradient(sf, laplacian_mask)
cv2.imwrite(f'{outfile_save_path}sf_laplacian.jpg', sf_laplacian)

#allow negative values in array
sf_laplacian = sf_laplacian.astype(np.int64)
laplacian_sharpened_sf = add_images(sf, -1*sf_laplacian)
cv2.imwrite(f'{outfile_save_path}laplacian_sharpened_sf.jpg', laplacian_sharpened_sf)