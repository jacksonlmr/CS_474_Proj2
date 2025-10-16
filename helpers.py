import numpy as np
from typing import Callable

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
    >Function that take 2 np.ndarray's as arguments and returns a np.ndarray 

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
    output_array = np.zeros((input_row, input_col), dtype=np.uint64)
    for current_row in range(input_row):
        for current_col in range(input_col):
            padded_row = current_row+pad_size
            padded_col = current_col+pad_size

            neighborhood = getNeighborhood(padded_img_array, (padded_row, padded_col), mask_size)
            pixel_value = operation(neighborhood, weights)

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