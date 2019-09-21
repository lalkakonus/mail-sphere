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

def test_score(dataset):
    # TEST dataset on sklearn
    return
    n_estimators = 100
    model = SklearnGradientBoostingClassifier(n_estimators=n_estimators)
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    prediction = np.empty([n_estimators, X_test.shape[0]])
    for i, proba in enumerate(model.staged_predict_proba(X_test)):
        prediction[i] = proba[:, 1]
    y = (- y_test * np.log(prediction) - (1 - y_test) * np.log(1 - prediction)).sum(axis=1)

    board = DashBoard()
    board.init_graph(name="baseline", title="sklearn", line_type="baseline", c="g", y=y.reshape(-1))
    board.make_plot()

def test_my_score(dataset):
    return
    n_estimators = 100
    model = GradientBoostingClassifier(n_estimators=n_estimators)
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    prediction = model.staged_predict_proba(X_train)
    y = (- y_test * np.log(prediction) - (1 - y_test) * np.log(1 - prediction)).sum(axis=1)

    board = DashBoard()
    board.init_graph(name="my_one", title="my_one", line_type="default", c="r", y=y.reshape(-1))
    board.make_plot()
