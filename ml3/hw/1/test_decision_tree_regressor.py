import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from decision_tree_regressor import DecisionTreeRegressor
from sklearn import datasets
from dataloader import DataLoader
from time import time
import numpy as np

@pytest.fixture
def dataset():
    """
    Load diabetes dataset.

    Return
    ------
    X_train, X_test, y_train, y_test - splitted dataset
    """
    test_dataset = datasets.load_diabetes()
    X = test_dataset.data
    y = test_dataset.target
    return train_test_split(X, y, test_size=0.33, random_state=42)

def test_regressor(dataset):
    X_train, X_test, y_train, y_test = dataset
    my_regressor = DecisionTreeRegressor(max_depth=1)
    my_regressor.fit(X_train, y_train)
    print(my_regressor.score(X_test, y_test))

def test_highload():
    X_train, X_test, y_train, y_test = DataLoader().load()
    
    regressor = SklearnDecisionTreeRegressor(max_depth=20)
    t0 = time()
    regressor.fit(X_train, y_train)
    print("Sklearn time: {}".format(time() - t0))
    print("Sklearn score: {}".format(regressor.score(X_test, y_test)))
    
    my_regressor = DecisionTreeRegressor(max_depth=20)
    t0 = time()
    my_regressor.fit(X_train, y_train)
    print("My time: {}".format(time() - t0))
    print(np.isfinite(X_test).all())
    #print(np.isfinite(my_regressor.predict(X_test)).all())
    #print("My score: {}".format(my_regressor.score(X_test, y_test))) 
