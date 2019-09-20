from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from decision_tree_regressor import DecisionTreeRegressor
from sklearn import datasets
import numpy as np
from time import time

test_dataset = datasets.load_diabetes()
X = test_dataset.data[:]
y = test_dataset.target[:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


my_regressor = DecisionTreeRegressor(min_samples_leaf=1)
t1 = time()
my_regressor.fit(X_train, y_train)
print(time() - t1)
# my_regressor.predict(X_test)
# print(my_regressor.score(X_test, y_test))

regressor = SklearnDecisionTreeRegressor(min_samples_leaf=1)
t1 = time()
regressor.fit(X_train, y_train)
print(time() - t1)
# my_regressor.predict(X_test)
# print(regressor.score(X_test, y_test))
