from PIL import Image
import numpy as np
import time
from numba import jit, cuda

outfile_save_path = "Output_Images/"
image = Image.open("Input_Images/Image.pgm")
image.save("Input_Images/Image.jpg")
pattern = Image.open("Input_Images/Pattern.pgm")
pattern.save("Input_Images/Pattern.jpg")

#create and save test image
test_img_pixels = np.array([[0, 0, 0, 0, 0], 
                            [0, 255, 255, 255, 0],
                            [0, 255, 255, 255, 0],
                            [0, 255, 255, 255, 0],
                            [0, 0, 0, 0, 0]], dtype=np.uint8)
test_img = Image.fromarray(test_img_pixels, 'L')
test_img.save("Input_Images/test_img.png")
test_img = Image.open("Input_Images/test_img.png")


def correlation(input_img_array: np.ndarray, weights: np.ndarray):
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
            # print(f'Neighborhood at ({current_row}, {current_col}): \n{neighborhood})')
            pixel_value = weightSumMatrix(neighborhood, weights)
            # print(f'Pixel value: {pixel_value}\n')

            output_array[current_row, current_col] = pixel_value

    # print(output_array)
    output_array = mapValues(output_array)
    # print(output_array)
    # output_img = Image.fromarray(obj = output_array, mode = 'L')
    # print(f"Output image in function: \n{np.array(output_img)}")

    return output_array

def getNeighborhood(input_img_array: np.ndarray, pixel: tuple, size: int):
    """
    Gets the neighborhood surrounding pixel

    **Parameters**
    ---------------
    >**input_img**:
    >numpy ndarray representing image

    >**pixel**:
    >Tuple containing (row, column) coordinates of center of the neighborhood

    >**size**:
    >equal to the width and height (neighborhood is always square) of the desired output

    **Returns**
    -----------
    >**neighborhood**: 2D array of size (size, size)
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
    sum = 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            sum += matrix[row, col]*weight[row, col]

    return int(sum)

def mapValues(input_img_array: np.ndarray):
    input_row, input_col = input_img_array.shape
    output_img_array = np.zeros((input_row, input_col), dtype=np.uint8)

    max_value = np.max(input_img_array)

    for current_row in range(input_row):
        for current_col in range(input_col):
            current_value = input_img_array[current_row, current_col]
            mapped_value = int(max(0, min(255, 255*(current_value/max_value))))
            output_img_array[current_row, current_col] = mapped_value

    return output_img_array

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
# correlation_test_array = np.array([
#     [0, 1, 2],
#     [2, 1, 0],
#     [0, 1, 2]
# ], dtype=np.uint8)

test_weights = np.ones((15, 15), dtype=np.uint8)

#using pattern array for weights
pattern_array = np.array(pattern, dtype=np.uint8)
zero_pattern = np.zeros((pattern_array.shape[1]-pattern_array.shape[0], pattern_array.shape[1]))
pattern_array = np.vstack((pattern_array, zero_pattern))
print(pattern_array.shape)

image_array = np.array(image, dtype=np.uint8)
start_time = time.time()
correlated_image_array = correlation(image_array, pattern_array)
end_time = time.time()
print(f'Time: {end_time-start_time}')

correlated_image = Image.fromarray(correlated_image_array)
correlated_image.save(f"{outfile_save_path}correlated_image.jpg")
