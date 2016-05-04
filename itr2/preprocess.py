__author__ = 'luoshalin'

import os
import numpy as np
import cv2
import fnmatch
import pandas as pd
from pyspark.mllib.linalg import Vectors

STANDARD_SIZE = 2700, 1800


def main():
    f_dir = '../data/sample/'
    label_fpath = '../data/trainLabels.csv'
    output_fpath = '../output/output.csv'
    pca_dim = 5     # how many features pca will maintain for each img
    dim = (2700, 1800)

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

    # get label dict
    label_dic = get_label_dic(label_fpath)  # <str, str>

    # get feature for all img files
    with open(output_fpath, "a") as f:
        for i in range(len(img_fpath_list)):
            img_fpath = img_fpath_list[i]
            img_name = img_name_list[i]
            # read
            org_img = cv2.imread(img_fpath)
            # resize
            img = resize_img(org_img, dim)  # crop the black space

            # get stats feature
            raw = img.flatten()
            (means, stds) = cv2.meanStdDev(img)
            stats = np.concatenate([means, stds]).flatten()  # np array

            # hog
            # hog = cv2.HOGDescriptor()
            # h = hog.compute(img)
            # print 'hog:'
            # print h.shape
            l = stats.tolist()
            l = [str(i) for i in l]
            feature_line = ','.join(l)

            # get label
            label = label_dic[img_name]

            # output
            line = label + ',' + feature_line + '\n'
            f.write(line)


def resize_img(img, dim):
    # perform the actual resizing of the image and show it
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img


def get_label_dic(label_fpath):
    label_dic = dict()
    with open(label_fpath) as f:
        line = f.readline().strip()
        while line != '':
            label_dic[line.split(',')[0]] = line.split(',')[1]
            line = f.readline().strip()
    return label_dic


def get_df(fpath):
    col_list = ['label', 'features']
    f = open(fpath)

    df_list = []
    with open(fpath) as f:
        line = f.readline().strip()
        while line != '':
            line_list = line.split(',')
            label = float(line_list[0])
            features = Vectors.dense([float(k) for k in line_list[1:]])
            df_list.append([label, features])
            line = f.readline().strip()

    df = pd.DataFrame(df_list, columns=col_list)
    return df


if __name__ == '__main__':
    main()