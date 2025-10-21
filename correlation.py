from PIL import Image
import numpy as np
from helpers import traverseImage, weightSumMatrix
from typing import Callable

outfile_save_path = "Output_Images/Correlation/"

image = Image.open("Input_Images/Image.pgm")
image.save("Input_Images/Image.jpg")

pattern = Image.open("Input_Images/Pattern.pgm")
pattern.save("Input_Images/Pattern.jpg")


def correlation(input_img_array: np.ndarray, weights: np.ndarray):
    """
    Computes the correlation of an image with a given mask.

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

#create a square array of weights with the given pattern
pattern_array = np.array(pattern, dtype=np.uint8)
zero_pattern = np.zeros((pattern_array.shape[1]-pattern_array.shape[0], pattern_array.shape[1]))
pattern_array = np.vstack((pattern_array, zero_pattern))

#convert image to numpy array, perform correlation
image_array = np.array(image, dtype=np.uint8)
correlated_image_array = correlation(image_array, pattern_array)

#convert result back to image, save image
correlated_image = Image.fromarray(correlated_image_array)
correlated_image.save(f"{outfile_save_path}correlated_image.jpg")



