import numpy as np
from typing import Callable
import random
from multipledispatch import dispatch

@dispatch(np.ndarray, np.ndarray, object)
def traverseImage(input_img_array: np.ndarray, weights: np.ndarray, operation: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    """
    Computes the correlation of an image with a given mask.

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image. Should have dtype=np.uint8.

    >**weights**:
    >np.ndarray representing the mask to be used. Should have dtype=np.uint8.

    >**operation**:
    >Function that take 2 np.ndarray's as arguments and returns an integer value for pixel value

    **Returns**
    -----------
    >**output_array**: 2D array representing the output of performing the passed functionality at each pixel
    """
    input_row, input_col = input_img_array.shape

    #determine padding size and pad image
    weights = np.array(weights)
    mask_size = weights.shape[1] #since mask should always be square
    pad_size = mask_size//2
    padded_img_array = np.pad(array=input_img_array, pad_width=pad_size)

    # height for rows, width for cols
    output_array = np.zeros((input_row, input_col), dtype=np.int64)
    for current_row in range(input_row):
        for current_col in range(input_col):
            padded_row = current_row+pad_size
            padded_col = current_col+pad_size

            neighborhood = getNeighborhood(padded_img_array, (padded_row, padded_col), mask_size)
            pixel_value = operation(neighborhood, weights)

            output_array[current_row, current_col] = pixel_value

    output_array = mapValues(output_array)
    return output_array

@dispatch(np.ndarray, int, object)
def traverseImage(input_img_array: np.ndarray, size: int, operation: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    """
    Computes the correlation of an image with a given mask.

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image. Should have dtype=np.uint8.

    >**size**:
    >size of the neighborhood to compute at each pixel

    >**operation**:
    >Function that takes 1 np.ndarray and size as arguments and returns an integer value for pixel value

    **Returns**
    -----------
    >**output_array**: 2D array representing the output of performing the passed functionality at each pixel
    """
    input_row, input_col = input_img_array.shape

    #determine padding size and pad image
    pad_size = size//2
    padded_img_array = np.pad(array=input_img_array, pad_width=pad_size)

    # height for rows, width for cols
    output_array = np.zeros((input_row, input_col), dtype=np.uint64)
    for current_row in range(input_row):
        for current_col in range(input_col):
            padded_row = current_row+pad_size
            padded_col = current_col+pad_size

            neighborhood = getNeighborhood(padded_img_array, (padded_row, padded_col), size)
            pixel_value = operation(neighborhood)

            output_array[current_row, current_col] = pixel_value

    output_array = mapValues(output_array)
    return output_array

def getNeighborhood(input_img_array: np.ndarray, pixel: tuple, size: int):
    """
    Gets the neighborhood surrounding pixel

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image

    >**pixel**:
    >Tuple containing (row, column) coordinates of the pixel to use for computing neighborhood

    >**size**:
    >equal to the width and height (neighborhood is always square) of the desired neighborhood shape

    **Returns**
    -----------
    >**neighborhood**: 2D numpy array of shape (size, size)
    """
    #straight line distance from center pixel to edge of neighborhood
    neighbor_distance = size//2
    
    #row and column position for top left corner of neighborhood in input_img
    top_left_row = pixel[0] - neighbor_distance
    top_left_col = pixel[1] - neighbor_distance
     
    #size of input image array
    input_row, input_col = input_img_array.shape
    
    neighborhood = np.zeros((size, size), dtype=np.uint8)

    for current_row, n_current_row in zip(range(top_left_row, top_left_row+size), range(size)):
        for current_col, n_current_col in zip(range(top_left_col, top_left_col+size), range(size)):
            
            #check to make sure coordinate is in bounds
            if 0 <= current_row < input_row and 0 <= current_col < input_col:
                #getpixel takes (width, height) -> (col, row)
                neighborhood[n_current_row, n_current_col] = input_img_array[current_row, current_col]
            else:
                neighborhood[n_current_row, n_current_col] = 0
    
    return neighborhood

def mapValues(input_img_array: np.ndarray):
    """
    Maps values from detected range to [0, 255]

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image

    **Returns**
    -----------
    >**output_img_array**: np.ndarray representing an image with the values mapped to [0, 255]
    """
    input_row, input_col = input_img_array.shape
    output_img_array = np.zeros((input_row, input_col), dtype=np.uint8)

    max_value = np.max(input_img_array)

    for current_row in range(input_row):
        for current_col in range(input_col):
            current_value = input_img_array[current_row, current_col]
            mapped_value = int(max(0, min(255, 255*(current_value/max_value))))
            output_img_array[current_row, current_col] = mapped_value

    return output_img_array

def weightSumMatrix(matrix: np.ndarray, weight: np.ndarray):
    """
    Sums all values of **matrix**, each weighted with the corresponding value in **weight**

    **Parameters**
    ---------------
    >**matrix**:
    >np.ndarray representing the matrix to be summed

    >**weight**:
    >np.ndarray representing the weights

    **Returns**
    -----------
    >**sum**: integer result of the operation
    """
    sum = 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            sum += matrix[row, col]*weight[row, col]

    return int(sum)

def salt_pepper_noise(input_img_array: np.ndarray, noise_percent: float):
    #calculate step to go through array based on percent
    input_rows = input_img_array.shape[0]
    input_cols = input_img_array.shape[1]
    output_img_array = input_img_array.copy()

    #at each, randomly make the pixel black or white (50% chance of each)
    for row in range(input_rows):
        for col in range(input_cols):
            if (random.random() < noise_percent):
                if random.randint(0, 1) == 0:
                    output_img_array[row, col] = 0
                else:
                    output_img_array[row, col] = 255

    return output_img_array

def get_median(input_array: np.ndarray):
    return int(np.median(input_array))

def add_images(input_img_1: np.ndarray, input_image_2: np.ndarray):
    """
    Computes the addition of 2 images. Input images should be the same shape. 

    **Parameters**
    ---------------
    >**x_values**:
    >np.ndarray. Should have dtype=np.uint8.

    >**y_values**:
    >np.ndarray. Should have dtype=np.uint8.

    **Returns**
    -----------
    >**output_array**: 2D array representing the addition of the 2 images.
    """
    #both images should be same shape
    rows, cols = input_img_1.shape

    #calculate the magnitude of the gradient at every pixel in the image
    summed_array = np.zeros(shape=(rows, cols), dtype=np.int64)
    for row in range(rows):
        for col in range(cols):
            img_1_value = input_img_1[row, col]
            img_2_value = input_image_2[row, col]

            values_sum = img_1_value+img_2_value
            summed_array[row, col] = values_sum

    summed_array = mapValues(summed_array)
    return summed_array
#test get median
# get_median_test_array = np.array([
#     [0, 1, 2, 3, 4],
#     [6, 7, 8, 9, 10],
#     [11, 12, 13, 14, 15],
#     [16, 17, 18, 19, 20],
#     [21, 22, 23, 24, 25]
# ])

# print(get_median(get_median_test_array))


#test salt and pepper noise            
# salt_pepper_test_array = np.array([
#     [0, 1, 2, 3, 4],
#     [6, 7, 8, 9, 10],
#     [11, 12, 13, 14, 15],
#     [16, 17, 18, 19, 20],
#     [21, 22, 23, 24, 25]
# ])
# print(salt_pepper_noise(salt_pepper_test_array, .5))

#test padding function
# padding_test_array = np.array([
#     [0, 1, 2, 3, 4],
#     [6, 7, 8, 9, 10],
#     [11, 12, 13, 14, 15],
#     [16, 17, 18, 19, 20],
#     [21, 22, 23, 24, 25]
# ])
# print(np.pad(padding_test_array, pad_width=1))
# padded_image = pad_0_img(image, 20)
# padded_image.save(f"{outfile_save_path}padded_image.jpg")

#test getNeighborhood
# neighborhood_test_array = np.array([
#     [0, 1, 2, 3, 4],
#     [6, 7, 8, 9, 10],
#     [11, 12, 13, 14, 15],
#     [16, 17, 18, 19, 20],
#     [21, 22, 23, 24, 25]
# ])

# neighborhood = getNeighborhood(neighborhood_test_array, (1, 3), 3)
# print(neighborhood)
# neighborhood = getNeighborhood(image, (240, 145), 100)
# image_neighborhood = Image.fromarray(neighborhood, 'L') 
# image_neighborhood.save(f"{outfile_save_path}image_test_neighborhood.jpg")

#test weightSumMatrix 
# m1 = np.array([
#     [0,0,0],
#     [1,1,1],
#     [2,2,2]
# ])

# m2 = np.array([
#     [2,2,2],
#     [2,2,2],
#     [2,2,2]
# ])

# matrix_weight_test_sum = weightSumMatrix(m1, m2)
# print(matrix_weight_test_sum)

#test value map
# map_test_array = np.array([
#     [300, 400, 500, 1000],
#     [1, 4, 8, 10],
#     [0, 0, 0, 0]
# ])
# print(mapValues(map_test_array))

# #test correlation