import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier
from gradient_boosting import GradientBoostingClassifier
from sklearn import datasets
from dataloader import DataLoader
from time import time
import numpy as np
from dashboard import DashBoard

@pytest.fixture
def dataset():
    """
    Load diabetes dataset.

    Return
    ------
    X_train, X_test, y_train, y_test - splitted dataset
    """
    test_dataset = datasets.load_breast_cancer()
    X = test_dataset.data
    y = test_dataset.target
    return train_test_split(X, y, test_size=0.33, random_state=42)

def test_classifier(dataset):
    return
    X_train, X_test, y_train, y_test = dataset
    my_classifier = GradientBoostingClassifier(max_depth=10, n_estimators=55)
    my_classifier.fit(X_train, y_train, X_test, y_test)
    print(my_classifier.score(X_test, y_test))

    classifier = SklearnGradientBoostingClassifier(max_depth=10, n_estimators=55)
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))
    # print(my_classifier.score(X_test, y_test))

'''
def test_score(dataset):
    # TEST dataset on sklearn
    return
    n_estimators = 300
    model = SklearnGradientBoostingClassifier(n_estimators=n_estimators)
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    prediction = np.empty([n_estimators, X_train.shape[0]])
    for i, proba in enumerate(model.staged_predict_proba(X_train)):
        prediction[i] = proba[:, 1]
    y = (- y_train * np.log(prediction) - (1 - y_train) * np.log(1 - prediction)).sum(axis=1)

    board = DashBoard()
    board.init_graph(name="baseline", title="sklearn", line_type="baseline", c="g", y=y.reshape(-1))
    board.make_plot()
'''

def loss(y_true, y_pred):
    eps = 1e-7
    return (- y_true * np.log(y_pred + eps) - (1 - y_true) * np.log(1 - y_pred + eps)).sum(axis=1)


def test_real_data(dataset):
    # X_train, X_valid, X_test, y_train, y_valid, y_test = DataLoader().load()
    X_train, X_test, y_train, y_test = DataLoader().load()
    # X_train, X_test, y_train, y_test = dataset
    #X_train_ = np.copy(X_train)
    #y_train_ = np.copy(y_train)
    #X_test_ = np.copy(X_test)
    #y_test_ = np.copy(y_test)

    n_estimators = 300
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=1,
        min_samples_split=20,
        subsample=0.5,
        learning_rate=0.2,
        #max_features=0.8,
        min_samples_leaf=10)
    model.fit(X_train, y_train.reshape(-1))
    prediction = model.staged_predict_proba(X_test)
    y = loss(y_test.reshape(-1), prediction)
    board = DashBoard()
    board.init_graph(name="my_one", title="my_one", line_type="default", c="r", y=y.reshape(-1))
   
    model = SklearnGradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=1,
        criterion="mse",
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.5,
        learning_rate=0.1,
        #max_features=0.1,
        init="zero")
    model.fit(X_train, y_train)
    prediction = np.empty([n_estimators, X_test.shape[0]])
    for i, proba in enumerate(model.staged_predict_proba(X_test)):
        prediction[i] = proba[:, 1]
    print(y_test.shape, prediction.shape)
    # assert 0
    y = loss(y_test.reshape(-1), prediction)

    board.init_graph(name="baseline", title="sklearn", line_type="baseline", c="g", y=y.reshape(-1))
    board.make_plot()
