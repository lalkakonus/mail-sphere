from sklearn.tree import DecisionTreeRegressor
import math

class ConstModel:
    def __inti__(self, target):
        self.target = target

    def predict(self, X):
        return np.ones(X.shape[0]) * self.target

class GradientBoostingClassifier:
    estimators_array = []
    
    def __inti__(self, learning_rate=0.1, n_estimators=100, subsample=1.0, 
            criterion="friedman_mse", min_samples_split=2, min_samples_leaf=1,
            max_depth=3, _iter_no_change=None):
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
            
        Return
        -------
        object : self
        """
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion=criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        return self

    def fit(self, X: np.array, y: np.array):
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
        self.initialize_model(X, y)
        for i in range(self.n_estimators):
            self.add_estimator(X, y)
        return self

    def predict_proba(self, X):
        result = np.zeros(X.shape[0])
        for model in self.estimator_array:
            result += model.predict(X)
        return result

    def predict(self, X):
        return self.predict_proba.round()

    def initialize_model(self, X, y):
        estimators_array += ConstMode(-math.log((1 / y.mean() - 1)))
    
    def add_estimator(self, X, y, depth=1):
        shift = 1 / (1 + np.exp(-self.predict_proba(X))) - y
        model = DecisionTreeRegression(depth=depth)
        model.fit(X, shift)
        
