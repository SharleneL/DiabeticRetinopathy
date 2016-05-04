__author__ = 'luoshalin'

from sklearn.decomposition import RandomizedPCA


# dimensionality reduction
# get <#img * n_component> np arr
def get_pca(arr, dim):
    pca = RandomizedPCA(n_components=dim)  # 5 dimension array
    train_x = pca.fit_transform(arr)
    return train_x