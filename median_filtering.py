from helpers import salt_pepper_noise
import cv2
import numpy as np

outfile_save_path = "Output_Images/"

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

# def median_filter(input_img_array: np.ndarray, size: int):