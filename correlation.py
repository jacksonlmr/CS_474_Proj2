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

# def correlation(input_img: Image, weights: np.ndarray):
#     input_img_array = np.array(input_img)
#     input_row, input_col = input_img_array.shape

#     #determine padding size and pad image
#     mask_size = weights.shape[0]
#     pad_size = mask_size//2
#     padded_img_array = pad_0_array(input_img_array, pad_size)

#     output_array = np.zeros_like(input_img_array, dtype=np.int64) 
#     for current_row in range(input_row):
#         print(f"row: {current_row}")
#         for current_col in range(input_col):
#             padded_row = current_row+pad_size
#             padded_col = current_col+pad_size
#             # current_input_pixel = padded_img.getpixel((padded_y, padded_x))
#             neighborhood = getNeighborhood(padded_img_array.flatten(), input_row, (padded_row, padded_col), mask_size)
#             pixel_value = weightSumMatrix(neighborhood, weights.flatten())
#             output_array[current_row, current_col] = pixel_value

#     output_array = mapValues(output_array)
#     output_img = Image.fromarray(obj = output_array, mode = 'L')

#     return output_img

def correlation(input_img: Image, weights: np.ndarray):
    input_img_array = np.array(input_img)
    input_row, input_col = input_img_array.shape

    #determine padding size and pad image
    mask_size = weights.shape[0]
    pad_size = mask_size//2
    padded_img_array = pad_0_array(input_img_array, pad_size)
    padded_img_width = padded_img_array.shape[1 ]
    padded_img_array = padded_img_array.flatten()

    output_array = np.zeros_like(input_img_array, dtype=np.int64).flatten()
    for pixel_index in range(input_row*input_col):
        pixel_row_col = (pixel_index//padded_img_width, pixel_index%padded_img_width)
        neighborhood = getNeighborhood(input_img_array=padded_img_array, input_img_width=padded_img_width, pixel=pixel_row_col, size=mask_size)
        pixel_value = weightSumMatrix(matrix=neighborhood, weights=weights)
        output_array[pixel_index] = pixel_value

    output_array = mapValues(output_array.reshape((input_row, input_col)))
    output_img = Image.fromarray(obj = output_array, mode = 'L') 

    return output_img

def pad_0_img(input_img: Image, pad_size: int):
    input_x, input_y = input_img.size
    padded_y = input_y + 2*pad_size
    padded_x = input_x + 2*pad_size

    padded_img = Image.new(mode = 'L', size = (padded_x, padded_y), color = 0)
    padded_img.paste(input_img, (pad_size, pad_size))

    return padded_img

# def getNeighborhood(input_img: Image, pixel: tuple, size: int):
#     neighbor_distance = size//2
    
#     #x and y coordinates for top left pixel of neighborhood
#     top_left_x = pixel[0] - neighbor_distance
#     top_left_y = pixel[1] - neighbor_distance
    
#     input_x, input_y = input_img.size
#     neighborhood = np.zeros((size, size), dtype=np.uint8)
#     for current_y, n_current_y in zip(range(top_left_y, top_left_y+size), range(size)):
#         for current_x, n_current_x in zip(range(top_left_x, top_left_x+size), range(size)):
            
#             #check to make sure coordinate is in bounds
#             if 0 <= current_y < input_y and 0 <= current_x < input_x:
#                 neighborhood[n_current_y, n_current_x] = input_img.getpixel((current_x, current_y))
#             else:
#                 neighborhood[n_current_y, n_current_x] = 0
    
#     return neighborhood

def getNeighborhood(input_img_array: np.ndarray, input_img_width: int, pixel: tuple, size: int):
    neighbor_distance = size//2
    
    #row and column coordinates for top left pixel of neighborhood
    top_left_row = pixel[0] - neighbor_distance
    top_left_col = pixel[1] - neighbor_distance

    neighborhood = np.zeros(size*size, dtype=input_img_array.dtype)

    #2d representation of ending column for the neighborhood
    end_col = top_left_col + size 
    # i represents the ith row of the neighborhood array
    for i in range(size):
        neighbor_start_index = i*size
        neighbor_end_index = neighbor_start_index + size
        #2d representation of row in matrix
        current_img_row = top_left_row + i

        #calculate indices for 1d image array
        img_start_index = current_img_row*input_img_width + top_left_col 
        img_end_index = current_img_row*input_img_width + end_col

        neighborhood[neighbor_start_index:neighbor_end_index] = input_img_array[img_start_index:img_end_index]
    
    return neighborhood

def weightSumMatrix(matrix: np.ndarray, weight: np.ndarray):
    sum = 0
    for i in range(matrix.size):
        sum += matrix[i]*weight[i]

    return int(sum)

def mapValues(input_img_array: np.ndarray):
    input_row, input_col = input_img_array.shape
    output_img_array = np.zeros_like(input_img_array, dtype=np.uint8)

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

#test getNeighborhood
# neighborhood_test_array = np.array([
#     [0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 0],
#     [0, 1, 1, 1, 0],
#     [0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0]
# ])
# neighborhood = getNeighborhood(neighborhood_test_array.flatten(), 5, (1, 1), 3)
# print(neighborhood.reshape((3, 3)))
# neighborhood = getNeighborhood(np.array(image), (240, 145), 100)
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

# matrix_weight_test_sum = weightSumMatrix(m1.flatten(), m2.flatten())
# print(matrix_weight_test_sum)

#test value map
# map_test_array = np.array([
#     [300, 400, 500, 1000],
#     [1, 4, 8, 10],
#     [0, 0, 0, 0]
# ])
# print(mapValues(map_test_array))
# #test correlation
# correlation_weights = np.full(shape = (7, 7), fill_value = 1)
pattern_array = np.array(pattern)
print(pattern_array.shape)
# zero_pattern = np.zeros((pattern_array.shape[1]-pattern_array.shape[0], pattern_array.shape[1]))
# pattern_array = np.vstack((pattern_array, zero_pattern))
# print(pattern_array.shape)
correlated_image = correlation(image, pattern_array)
correlated_image.save(f"{outfile_save_path}correlated_image.jpg")
