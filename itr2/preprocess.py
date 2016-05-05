__author__ = 'luoshalin'

import os
import numpy as np
import cv2
import fnmatch

STANDARD_SIZE = 2700, 1800


def main():
    f_dir = '../data/train/'
    label_fpath = '../data/trainLabels.csv'
    output_fpath = '../output/1000_feature_output.csv'
    pca_dim = 5     # how many features pca will maintain for each img
    dim = (2700, 1800)
    sample_amt = 1000

    # get all img file paths
    img_fpath_list = []
    img_name_list = []
    i = 0
    for root, dirs, files in os.walk(f_dir):
        for file in files:
            print 'read in file#' + str(i)
            if i >= sample_amt:
                break
            img_name_list.append(file)
            fpath = os.path.join(root, file)
            img_fpath_list.append(fpath)
            i += 1
        if i >= sample_amt:
            break

    img_fpath_list = fnmatch.filter(img_fpath_list, '*.jpeg')
    img_name_list = fnmatch.filter(img_name_list, '*.jpeg')
    img_name_list = [f[:-5] for f in img_name_list]

    # get label dict
    label_dic = get_label_dic(label_fpath)  # <str, str>

    print 'l42'
    # get feature for all img files
    with open(output_fpath, "a") as f:
        for i in range(len(img_fpath_list)):
            print 'process line#' + str(i)
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





if __name__ == '__main__':
    main()