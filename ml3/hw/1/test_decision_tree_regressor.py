import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from decision_tree_regressor import DecisionTreeRegressor
from sklearn import datasets
from dataloader import DataLoader
from time import time
from sklearn.metrics import mean_squared_error
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
    return train_test_split(X, y, test_size=0.3, random_state=42)

def test_regressor(dataset):
    #X_train, X_test, y_train, y_test = dataset
    X_train, X_valid, X_test, y_train, y_valid, y_test = DataLoader().load()
    my_regressor = DecisionTreeRegressor(max_depth=1, min_samples_split=2, min_samples_leaf=1)
    my_regressor.fit(X_train, y_train)
    my_score = my_regressor.score(X_test, y_test)
    
    regressor = SklearnDecisionTreeRegressor(max_depth=1, min_samples_split=2, min_samples_leaf=1)
    regressor.fit(X_train, y_train)
    sklearn_score = mean_squared_error(y_test, regressor.predict(X_test))
   
    print(my_score)
    print("Sklearn - My: {}".format(sklearn_score - my_score))

def test_highload():
    return 0
    X_train, X_test, y_train, y_test = DataLoader().load()
   
    print(X_train[10])
    regressor = SklearnDecisionTreeRegressor(max_depth=1)
    t0 = time()
    regressor.fit(X_train, y_train)
    print("Sklearn time: {}".format(time() - t0))
    print("Sklearn score: {}".format(regressor.score(X_test, y_test)))
    
    my_regressor = DecisionTreeRegressor(max_depth=1)
    t0 = time()
    my_regressor.fit(X_train, y_train)
    print("My time: {}".format(time() - t0))
    print(np.isfinite(X_test).all())
    #print(np.isfinite(my_regressor.predict(X_test)).all())
    #print("My score: {}".format(my_regressor.score(X_test, y_test))) 
