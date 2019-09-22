from decision_tree_regressor import DecisionTreeRegressor
import math
import logging
from scipy.optimize import minimize
import numpy as np
from sklearn.metrics import accuracy_score

class ConstModel:
    def fit(self, target: float):
        self.target = target
        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predict constnt value for every sample

        Paramert
        --------
        X : np.array of shape [n_samples, n_features]
        """
        n_samples = X.shape[0]
        return np.ones(n_samples) * self.target

class GradientBoostingClassifier:
    # Models
    estimators_array = []

    # Train set predictions
    predictions_array = None
    
    # Train set accuracy values
    accuracy_array = []

    variance_array = [] 

    def __init__(self, learning_rate=0.1, n_estimators=100, subsample=1.0, 
            criterion="mse", min_samples_split=2, min_samples_leaf=1,
            max_depth=3, max_features=1.0):
        """
        Initialize gradient boosting classifier with logistic loss function

        Parametrs
        ---------
        learning_rate : float > 0
        n_etimators : int > 0
            Count of trees in ensamble.
        subsample : float in (0, 1]
            The fraction of samples to be used for fitting the individual base learners.
        criterion : String from {"mse"}
            Loss function to build tree models that approximate antigradient of error function
        max_depth : int > 1
            Depth of tree model is gradually increase from 1 ti max_depth value
        """
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion=criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self, X_train: np.array, y_train: np.array, X_test=None, y_test=None):
        """
        Train model using given dataset.

        Parametrs
        --------
        X : np.array of shape [n_samples, n_features]
        y : np.arrat of shape [n_samples]

        Return
        ------
        object : self
        """
        self.predictions_array = np.empty([self.n_estimators, X_train.shape[0]])
        
        self.initialize_model(X_train, y_train)
        depth = 1
        #last_increase = 0
        for i in range(self.n_estimators):
            #if i % 33 == 0:
            #if np.array(self.accuracy_array[-10:]).var() < 1e-4 and last_increase + 10 < i:
            #depth += 1
            #last_increase = i
            #    print("Increase depth")
            self.add_estimator(X_train, y_train, depth)
        return self

    def staged_predict_proba(self, X):
        n_samples = X.shape[0]
        result = np.zeros([self.n_estimators, n_samples])

        result[0] = self.estimators_array[0].predict(X).reshape(-1)
        result[0] += self.learning_rate * self.estimators_array[1].predict(X).reshape(-1)
        for i, estimator in enumerate(self.estimators_array[2:], 1):
            result[i] = result[i - 1] + self.learning_rate * estimator.predict(X).reshape(-1)
        return 1 / (1 + np.exp(-result))

    def predict_proba(self, X: np.array) -> np.array:
        """
        Predict class probabilities [0, 1] for 1 class in {0, 1} classification

        Parametrs
        ---------
        X : np.array of shape [n_samples, n_features]

        Return
        --------
        prediction : np.array of shape [n_samples]
            Probabilities of 1 class.
        """
        return self.staged_predict_proba(X)[-1]

    def predict(self, X: np.array):
        return self.predict_proba(X).round()

    #def score(self, X, y_true):
    #    y_pred = self.predict(X)
    #    return accuracy_score(y_true, y_pred)

    @property
    def estimators_cnt(self):
        return len(self.estimators_array)

    @property
    def last_prediction(self):
        margin = self.last_margin
        return 1 / (1 + np.exp(-margin))

    def initialize_model(self, X, y):
        model = ConstModel().fit(-math.log(1 / y.mean() - 1))
        self.estimators_array.append(model)
        self.last_margin = model.predict(X).reshape(-1)
        #self.accuracy_array.append(accuracy_score(y, self.last_prediction.round()))
    
    def add_estimator(self, X, y, depth):
        target = y - self.last_prediction
        model = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            max_features=self.max_features).fit(X, target)

        self.last_margin = self.last_margin + self.learning_rate * model.predict(X).reshape(-1)
       
        # self.accuracy_array.append(accuracy_score(y, self.last_prediction.round()))
        self.estimators_array.append(model)
    
    def print_statistic(self, X_train, y_train, X_test, y_test, iteration):
        print("Iteration # {:<3}- ".format(iteration) + "train score: " + str(self.score(X_train, y_train)) +
            " - test score: " + str(self.score(X_test, y_test)))
