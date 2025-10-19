from helpers import salt_pepper_noise, traverseImage, get_median
import cv2
import numpy as np

outfile_save_path = "Output_Images/Median_Filtering/"

lenna = cv2.imread('Input_Images/lenna.gif', flags=0)
boat = cv2.imread('Input_Images/boat.gif', flags=0)

#salt and pepper images
lenna_sp_30 = salt_pepper_noise(lenna, .3)
cv2.imwrite(f'{outfile_save_path}lenna_sp_30.jpg', lenna_sp_30)

lenna_sp_50 = salt_pepper_noise(lenna, .5)
cv2.imwrite(f'{outfile_save_path}lenna_sp_50.jpg', lenna_sp_50)

boat_sp_30 = salt_pepper_noise(boat, .3)
cv2.imwrite(f'{outfile_save_path}boat_sp_30.jpg', boat_sp_30)

boat_sp_50 = salt_pepper_noise(boat, .5)
cv2.imwrite(f'{outfile_save_path}boat_sp_50.jpg', boat_sp_50)

def median_filter(input_img_array: np.ndarray, size: int):
    """
    Performs median filtering on an image to remove salt and pepper noise

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image. Should have dtype=np.uint8.

    >**size**:
    >int representing the size of the averaging mask

    **Returns**
    -----------
    >**output_array**: 2D array representing the filtered image
    """
    return traverseImage(input_img_array=input_img_array, size=size, operation=get_median)

#lenna filtering
lenna_sp_30_median_filtered7 = median_filter(lenna_sp_30, 7)
cv2.imwrite(f'{outfile_save_path}lenna_sp_30_median_filtered7.jpg', lenna_sp_30_median_filtered7)

lenna_sp_30_median_filtered15 = median_filter(lenna_sp_30, 15)
cv2.imwrite(f'{outfile_save_path}lenna_sp_30_median_filtered15.jpg', lenna_sp_30_median_filtered15)

lenna_sp_50_median_filtered7 = median_filter(lenna_sp_50, 7)
cv2.imwrite(f'{outfile_save_path}lenna_sp_50_median_filtered7.jpg', lenna_sp_50_median_filtered7)

lenna_sp_50_median_filtered15 = median_filter(lenna_sp_50, 15)
cv2.imwrite(f'{outfile_save_path}lenna_sp_50_median_filtered15.jpg', lenna_sp_50_median_filtered15)


# boat filtering
boat_sp_30_median_filtered7 = median_filter(boat_sp_30, 7)
cv2.imwrite(f'{outfile_save_path}boat_sp_30_median_filtered7.jpg', boat_sp_30_median_filtered7)

boat_sp_30_median_filtered15 = median_filter(boat_sp_30, 15)
cv2.imwrite(f'{outfile_save_path}boat_sp_30_median_filtered15.jpg', boat_sp_30_median_filtered15)

boat_sp_50_median_filtered7 = median_filter(boat_sp_50, 7)
cv2.imwrite(f'{outfile_save_path}boat_sp_50_median_filtered7.jpg', boat_sp_50_median_filtered7)

boat_sp_50_median_filtered15 = median_filter(boat_sp_50, 15)
cv2.imwrite(f'{outfile_save_path}boat_sp_50_median_filtered15.jpg', boat_sp_50_median_filtered15)