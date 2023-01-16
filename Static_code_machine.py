import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


# pca计算
import Tool_io


def pca_sklearn(data):
    pca = PCA()
    new_data = pca.fit_transform(data)
    return new_data


# ridge计算
def ridge_sklearn(X_train, y_train,static_model,pro_name,ver_name):
    tar_path = os.path.join(static_model, pro_name)
    if not os.path.exists(tar_path):
        os.mkdir(tar_path)
    model = Tool_io.checkAndLoad(tar_path, pro_name + '_' + ver_name + '_static.mod')
    if model != None:
        return model
    model = Ridge(alpha=1.0)
    model = model.fit(X_train, y_train)
    name = os.path.join(tar_path, pro_name + '_' + ver_name + '_static.mod')
    with open(name, 'wb') as f:
        pickle.dump(model, f)
    # y_predication = model.predict(X_test)
    # res = np.array(y_train)
    # res = np.append(res,y_predication)
    return model


if __name__ == "__main__":
    mat = [
        [2, 2.7, 5.6],
        [2.0, 1.6, 4.2],
        [1.0, 1.1, 0.7],
        [1.5, 1.6, 8.7],
        [1.1, 0.9, 5.3],
        [2.5, 2.4, 4.3],
        [0.5, 0.7, 2.6],
        [2.2, 2.9, 3.5],
        [1.9, 2.2, 0.5],
        [3.1, 3., 2.1],
    ]
    data = np.array(mat)
    newData = pca_sklearn(data)
    y = [1, 9, 8, 7, 4, 5, 6, 3, 2, 0]
    ridge_sklearn(newData, y)
    print(newData)
