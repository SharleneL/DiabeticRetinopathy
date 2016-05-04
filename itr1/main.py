__author__ = 'luoshalin'

import os
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from preprocess import crop_img, resize_img, get_label_dic, get_label_m
from features.rgb import get_rgb, get_rgb_m
from features.gist import get_pca
import fnmatch

STANDARD_SIZE = 2700, 1800

def main():
    f_dir = '../../../resource/data/sample/'
    label_fpath = '../../../resource/data/trainLabels.csv'
    pca_dim = 5     # how many features pca will maintain for each img

    # get all img file paths
    img_fpath_list = []
    img_name_list = []
    for root, dirs, files in os.walk(f_dir):
        for file in files:
            img_name_list.append(file)
            fpath = os.path.join(root, file)
            img_fpath_list.append(fpath)

    img_fpath_list = fnmatch.filter(img_fpath_list, '*.jpeg')
    img_name_list = fnmatch.filter(img_name_list, '*.jpeg')
    img_name_list = [f[:-5] for f in img_name_list]

    # get feature for all img files
    rgb_m = []
    for img_fpath in img_fpath_list:
        # preprocess
        # cropped_img = crop_img(img_fpath)  # crop the black space
        # cropped_img.show()
        img = Image.open(img_fpath)
        resized_img = resize_img(img, STANDARD_SIZE)

        # get features
        img_rgb_arr = get_rgb(resized_img)  # np array of all pixels in the img; <#px, 3>
        img_rgb_arr = get_rgb_m(img_rgb_arr)  # convert <#px, 3> array into <1, #px*3> array. (rgbrgbrgb...)
        rgb_m.append(img_rgb_arr)

    rgb_m = np.array(rgb_m)  # a <#img_num, #px_num*3> np array
    pca_feature_m = get_pca(rgb_m, pca_dim)  # a <#img_num, #pca_dim> np array

    # construct label matrix
    label_dic = get_label_dic(label_fpath)  # <str, str>
    label_m = get_label_m(label_dic, img_name_list)

    # KNN
    knn = KNeighborsClassifier()
    knn.fit(pca_feature_m, label_m)

    print(knn.predict([[1.1, 1.2, 3.4, 5.4, 2.1]]))



if __name__ == '__main__':
    main()