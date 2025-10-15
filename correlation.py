from PIL import Image
import numpy as np

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


def correlation(input_img: Image, weights: np.ndarray):
    input_width, input_height = input_img.size

    #determine padding size and pad image
    weights = np.array(weights)
    mask_size = weights.shape[1]
    pad_size = mask_size//2
    padded_img = pad_0_img(input_img, pad_size)
    # print(f"Padded image: \n{np.array(padded_img, dtype=np.uint8)}")

    # height for rows, width for cols
    output_array = np.zeros((input_height, input_width), dtype=np.uint64)
    for current_row in range(input_height):
        for current_col in range(input_width):
            padded_row = current_row+pad_size
            padded_col = current_col+pad_size
            # current_input_pixel = padded_img.getpixel((padded_y, padded_x))
            neighborhood = getNeighborhood(padded_img, (padded_row, padded_col), mask_size)
            # print(f"neighborhood at ({padded_row}, {padded_col}): \n{neighborhood}")
            pixel_value = weightSumMatrix(neighborhood, weights)
            output_array[current_row, current_col] = pixel_value

    output_array = mapValues(output_array)
    # print(output_array)
    output_img = Image.fromarray(obj = output_array, mode = 'L')
    # print(f"Output image in function: \n{np.array(output_img)}")

    return output_img

def pad_0_img(input_img: Image, pad_size: int):
    input_width, input_height = input_img.size
    padded_width = input_width + 2*pad_size
    padded_height = input_height + 2*pad_size

    padded_img = Image.new(mode = 'L', size = (padded_width, padded_height), color = 0)
    padded_img.paste(input_img, (pad_size, pad_size))

    return padded_img


def getNeighborhood(input_img: Image, pixel: tuple, size: int):
    """
    Gets the neighborhood surrounding pixel

    **Parameters**
    ---------------
    >**input_img**:
    >Image object, can be any size

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
     
    #two lines for clarity between libraries
    input_width, input_height = input_img.size
    input_row, input_col = input_height, input_width
    
    neighborhood = np.zeros((size, size), dtype=np.uint8)

    for current_row, n_current_row in zip(range(top_left_row, top_left_row+size), range(size)):
        for current_col, n_current_col in zip(range(top_left_col, top_left_col+size), range(size)):
            
            #check to make sure coordinate is in bounds
            if 0 <= current_row < input_row and 0 <= current_col < input_col:
                #getpixel takes (width, height) -> (col, row)
                neighborhood[n_current_row, n_current_col] = input_img.getpixel((current_col, current_row))
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
# padded_image = pad_0_img(image, 20)
# padded_image.save(f"{outfile_save_path}padded_image.jpg")

# #test getNeighborhood
# neighborhood = getNeighborhood(image, (240, 145), 100)
# image_neighborhood = Image.fromarray(neighborhood, 'L') 
# image_neighborhood.save(f"{outfile_save_path}image_test_neighborhood.jpg")

# #test weightSumMatrix 
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
# correlation_weights = np.full(shape = (3, 3), fill_value = 1)

# correlation_test_array = np.full(shape = (10, 10), fill_value = 1, dtype=np.uint8)

# correlation_test_array = np.array([
#     [0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1],
#     [2, 2, 2, 2, 2, 2],
#     [3, 3, 3, 3, 3, 3]
# ], dtype=np.uint8)
# correlation_test_img = Image.fromarray(correlation_test_array)
# print(correlation_weights)
# print(correlation_test_array)
# correlated_test_img = correlation(correlation_test_img, correlation_weights)

# print(np.array(correlated_test_img, dtype=np.uint8))
# correlated_image = correlation(image, correlation_weights)
# correlated_image.save(f"{outfile_save_path}correlated_image.jpg")

pattern_array = np.array(pattern, dtype=np.uint8)
zero_pattern = np.zeros((pattern_array.shape[1]-pattern_array.shape[0], pattern_array.shape[1]))
pattern_array = np.vstack((pattern_array, zero_pattern))
print(pattern_array.shape)

correlated_image = correlation(image, pattern_array)

correlated_image.save(f"{outfile_save_path}correlated_image.jpg")
