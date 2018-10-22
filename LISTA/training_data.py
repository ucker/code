# Get Training and Testing Data
import numpy as np
from dictionary import Wd
from CoD import CoD
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn import preprocessing

def sparse_coding():
    '''
    Returns:
        (data_train, label_train) : wine data and label for training
        np.array(sparse_code) : optimal sparse code
        opti_Wd : optimal dictionary
        (data_test, label_test) : wine data and label for testing
    '''
    data = load_wine()
    print("wine loaded...")
    data_train_pre, data_test_pre, label_train, label_test = \
    train_test_split(data['data'],data['target'],
                     test_size=0.4)
    print("Data standardization...")
    # Data standardization ensures fast convergence of CoD algorithm
    scaler = preprocessing.StandardScaler().fit(data_train_pre)
    data_train = scaler.transform(data_train_pre)
    data_test = scaler.transform(data_test_pre)
    sparse_code = []
    print("Learning dictionary...")
    opti_Wd = Wd(data_train, 100)
    print("Getting optimal sparse code...")
    for i in data_train:
        sparse_code.append(CoD(i, opti_Wd, 0.5))
    return ((data_train, label_train), np.array(sparse_code), opti_Wd, (data_test, label_test))