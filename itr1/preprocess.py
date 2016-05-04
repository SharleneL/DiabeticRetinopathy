__author__ = 'luoshalin'

from pylab import *
from PIL import Image
import PIL.ImageOps
import numpy as np
from PIL import Image, ImageChops, ImageOps


def crop_img(fpath):
    img = Image.open(fpath)
    inv_img = PIL.ImageOps.invert(img)  # invert image color

    pix = np.asarray(inv_img)

    pix = pix[:,:,0:3]  # Drop the alpha channel
    idx = np.where(pix-255)[0:2] # Drop the color when finding edges
    box = map(min,idx)[::-1] + map(max,idx)[::-1]

    region = img.crop(box)
    return region


def resize_img(img, std_size):
    # width, height = img.size
    # size = width * std_height / height, std_height  # convert all images with the same height
    resized_img = ImageOps.fit(img, std_size, Image.ANTIALIAS, (0.5, 0.5))
    # resized_img.show()
    return resized_img


def get_label_dic(label_fpath):
    label_dic = dict()
    with open(label_fpath) as f:
        line = f.readline().strip()
        while line != '':
            label_dic[line.split(',')[0]] = line.split(',')[1]
            line = f.readline().strip()
    return label_dic


def get_label_m(label_dic, img_name_list):
    label_m = []
    for img_name in img_name_list:
        label_m.append(label_dic[img_name])
    return np.array(label_m)