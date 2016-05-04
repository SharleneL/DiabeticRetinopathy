__author__ = 'luoshalin'

import numpy as np


# turn an img into a numpy array of RGB pixels
def get_rgb(img):
    img_px_list = list(img.getdata())
    img_px_map = map(list, img_px_list)
    img_px_arr = np.array(img_px_map)
    return img_px_arr


# convert a numpy array of RGB pixels into a 2D matrix
def get_rgb_m(img_rgb_arr):
    s = img_rgb_arr.shape[0] * img_rgb_arr.shape[1]
    img_wide = img_rgb_arr.reshape(1, s)
    return img_wide[0]