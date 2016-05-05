__author__ = 'luoshalin'

import pandas as pd
from pyspark.mllib.linalg import Vectors


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