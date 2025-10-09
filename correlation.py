from PIL import Image
import numpy as np

image = Image.open("Input_Images/Image.pgm")
image.save("Image.jpg")
pattern = Image.open("Input_Images/Pattern.pgm")
pattern.save("Pattern.jpg")

def resize(input_img: Image, width: int, height: int):
    input_width, input_height = input_img.size
    factor = width/input_width

    output_size = (width, height)
    output_img = Image.new(size=output_size, mode='L')

    for row in range(width):
        for col in range(height):
            input_pixel = input_img.getpixel((int(row/factor), int(col/factor)))
            output_img.putpixel((row, col), input_pixel)

    return output_img

#create and save test image
test_img_pixels = np.array([[0, 0, 0, 0, 0], 
                            [0, 255, 255, 255, 0],
                            [0, 255, 255, 255, 0],
                            [0, 255, 255, 255, 0],
                            [0, 0, 0, 0, 0]], dtype=np.uint8)
test_img = Image.fromarray(test_img_pixels, 'L')
test_img.save("test_img.png")
test_img = Image.open("test_img.png")

def correlation(input_img: Image, weights):
    output_img = Image.new(mode = 'L', size = input_img.size)
    input_cols, input_rows = input_img.size

    #determine padding size and pad image
    weights = np.array(weights)
    mask_size = weights.shape[0]
    pad_size = mask_size//2
    padded_img = pad_0_img(input_img, pad_size)

    for row in range(input_rows):
        for col in range(input_cols):
            padded_row = row+pad_size
            padded_col = col+pad_size
            current_input_pixel = input_img.getpixel((padded_row, padded_col))
            neighborhood = getNeighborhood(padded_img, current_input_pixel, mask_size)

    return output_img

def pad_0_img(input_img: Image, pad_size: int):
    input_cols, input_rows = input_img.size
    padded_rows = input_rows + 2*pad_size
    padded_cols = input_cols + 2*pad_size

    padded_img = Image.new(mode = 'L', size = (padded_cols, padded_rows), color = 0)
    padded_img.paste(input_img, (pad_size, pad_size))

    return padded_img

def getNeighborhood(input_img: Image, pixel: tuple, size: int):
    neighbor_distance = size//2
    
    #row and column coordinates for top left pixel of neighborhood
    top_left_col = pixel[0] - neighbor_distance
    top_left_row = pixel[1] - neighbor_distance
    
    input_cols, input_rows = input_img.size
    neighborhood = np.zeros((size, size), dtype=np.uint8)
    for row, nrow in zip(range(top_left_row, top_left_row+size), range(size)):
        for col, ncol in zip(range(top_left_col, top_left_col+size), range(size)):
            if 0 <= row < input_rows and 0 <= col < input_cols:
                neighborhood[nrow, ncol] = input_img.getpixel((col, row))
            else:
                neighborhood[nrow, ncol] = 0
    # print(neighborhood)
    return neighborhood

#test padding function
padded_image = pad_0_img(image, 20)
padded_image.save("padded_image.jpg")

#test getNeighborhood
neighborhood = getNeighborhood(image, (240, 145), 100)
print(neighborhood)
image_neighborhood = Image.fromarray(neighborhood, 'L')
image_neighborhood.save("image_test_neighborhood.jpg")


