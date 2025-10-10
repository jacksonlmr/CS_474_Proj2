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
    input_x, input_y = input_img.size

    #determine padding size and pad image
    weights = np.array(weights)
    mask_size = weights.shape[0]
    pad_size = mask_size//2
    padded_img = pad_0_img(input_img, pad_size)

    output_array = np.zeros_like(input_img, dtype=np.uint8)
    for current_y in range(input_y):
        for current_x in range(input_x):
            padded_x = current_x+pad_size
            padded_y = current_y+pad_size
            # current_input_pixel = padded_img.getpixel((padded_y, padded_x))
            neighborhood = getNeighborhood(padded_img, (padded_x, padded_y), mask_size)
            pixel_value = weightSumMatrix(neighborhood, weights)
            output_array[current_y, current_x] = pixel_value

    output_array = mapValues(output_array)
    output_img = Image.fromarray(obj = output_array, mode = 'L')

    return output_img

def pad_0_img(input_img: Image, pad_size: int):
    input_x, input_y = input_img.size
    padded_y = input_y + 2*pad_size
    padded_x = input_x + 2*pad_size

    padded_img = Image.new(mode = 'L', size = (padded_x, padded_y), color = 0)
    padded_img.paste(input_img, (pad_size, pad_size))

    return padded_img

def getNeighborhood(input_img: Image, pixel: tuple, size: int):
    neighbor_distance = size//2
    
    #x and y coordinates for top left pixel of neighborhood
    top_left_x = pixel[0] - neighbor_distance
    top_left_y = pixel[1] - neighbor_distance
    
    input_x, input_y = input_img.size
    neighborhood = np.zeros((size, size), dtype=np.uint8)
    for current_y, n_current_y in zip(range(top_left_y, top_left_y+size), range(size)):
        for current_x, n_current_x in zip(range(top_left_x, top_left_x+size), range(size)):
            
            #check to make sure coordinate is in bounds
            if 0 <= current_y < input_y and 0 <= current_x < input_x:
                neighborhood[n_current_y, n_current_x] = input_img.getpixel((current_x, current_y))
            else:
                neighborhood[n_current_y, n_current_x] = 0
    
    return neighborhood

def weightSumMatrix(matrix: np.ndarray, weight: np.ndarray):
    sum = 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            sum += matrix[row, col]*weight[row, col]

    return int(sum)

def mapValues(input_img_array: np.ndarray):
    input_x, input_y = input_img_array.shape
    output_img_array = np.zeros_like(input_img_array)

    max_value = np.max(input_img_array)

    for current_y in range(input_y):
        for current_x in range(input_x):
            current_value = input_img_array[current_x, current_y]
            mapped_value = int(max(0, min(255, 255*(current_value/max_value))))
            output_img_array[current_x, current_y] = mapped_value

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
# correlation_weights = np.full(shape = (7, 7), fill_value = 1)
# correlated_image = correlation(image, correlation_weights)
# correlated_image.save(f"{outfile_save_path}correlated_image.jpg")
