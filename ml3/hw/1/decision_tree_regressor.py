import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import abc
from sklearn.metrics import balanced_accuracy_score

class BaseDecisionTree(metaclass=abc.ABCMeta):
    
    root = None
    
    class Node():
        def __init__(self, feature_num=None, predicate_value=None, left_node=None, right_node=None, node_type="NONLEAF",
                     target_value=None):
            self.feature_num = feature_num
            self.predicate_value = predicate_value
            self.right_node = right_node
            self.left_node = left_node
            self.target_value = target_value
            self.node_type = node_type
        
        def predict(self, sample: np.ndarray) -> np.ndarray:
            if self.node_type == "LEAF":
                return self.target_value
            else:
                if sample[self.feature_num] <= self.predicate_value:
                    return self.left_node.predict(sample)
                else:
                    return self.right_node.predict(sample)

    @abc.abstractmethod
    def create_node(self):
        pass

    @abc.abstractmethod
    def find_best_split(self, X: np.ndarray, y: np.ndarray):
        pass

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass


class DecisionTreeRegressor(BaseDecisionTree):
    
    def __init__(self, criterion="mse", min_samples_split=3, min_samples_leaf=3, max_depth=None, subsample=1.0,
            max_features=1.0):
        """
        Initialize Decision tree regressor.

        Parameters
        ---------
        criterion: string, optional (deffailt="mse")
            The function to measure the quality of a split. Supported criteria
            are "mse" for the Mean Squared Error, which is equal to sum_i=0^N{y_pred - y_true}^2
        max_deptx: int
            Maximum avaliable depth.
        min_samples_split: int >= 0 
            Minimum count of samples in split to split it into.
        min_samples_leaf:  int >= 0
            Minimim cout of samples in split to make leaf node.
        """
        assert criterion in {"mse"}
        assert type(max_depth) is int and max_depth > 0 or max_depth is None
        assert type(min_samples_split) is int and min_samples_split > 0
        assert type(min_samples_leaf) is int and min_samples_leaf > 0
        assert type(subsample) is float and subsample <= 1 and subsample >= 0

        self.subsample = subsample
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    # TODO transform classes to 0 - N classes
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model using samples from X with target values y.
        
        Parametrs
        ---------
        X : nd.array of shape [n_samples, n_features] 
            Array of samples in current split.
        y : nd.arrat of shape [n_samples] 
            Array of samples label in current split.
        depth : int >= 0 
            Current node depth.
        
        Return
        ---------
        self : object 
        """
        self.n_features = X.shape[1]
        idx = np.random.choice(X.shape[0], round(X.shape[0] * self.subsample), replace=False)
        #features = np.random.choice(self.n_features, round(self.n_features * self.max_features), replace=False)
        self.root = self.create_node(X[idx], y[idx], depth=0)
        #self.root = self.create_node(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples from X.
        
        Parametrs
        ---------
        X : nd.array of shape [n_samples, n_features] 
            Array of samples.
        
        Return
        ---------
        targets : nd.array of shape [n_sample]
            Array of predicted target values for each sample
        """
        return np.array([self.root.predict(sample) for sample in X]).reshape(-1, 1)

    def score(self, X: np.ndarray, y_true: np.ndarray, metrics="mse") -> float:
        """
        Mean Squared Error score.
        
        Parametrs
        ---------
        X : nd.array of shape [n_samples, n_features] 
            Array of samples.
        y_true : nd.array of shape [n_samples] 
            Array of true target values.
        metrics : sting, valid values "mse" and "r2_score"

        Return
        ---------
        mse : float
        """
        valid_metrics = {"mse", "r2_score"}
        assert metrics in valid_metrics
        
        if metrics == "r2_score":
            return r2_score(y_true, self.predict(X))
        else:
            return mean_squared_error(y_true, self.predict(X))

    def create_node(self, X: np.ndarray, y: np.ndarray, depth: int) -> BaseDecisionTree.Node:
        """
        Find best split parametr e.g. feature and predicate value and create NONLEAF node
        in case it is possible, ether create LEAF node with target value.

        Parametrs
        ---------
        X : nd.array of shape [n_samples, n_features] 
            Array of samples in current split.
        y : nd.arrat of shape [n_samples] 
            Array of sample labels in current split.
        depth : int >= 0 
            Current node depth.
        
        Return
        ---------
        node : Node
            LEAF node (with target value attribute) or NONLEAF node (with feature and predicate value attributes)
        """
        if ((self.max_depth is not None and depth >= self.max_depth) or 
            y.size < max(self.min_samples_leaf * 2, self.min_samples_split * 2)):
            # if split is too small or branch deep enought, create leaf node
            target_value = 0.5 * y.sum() / (np.abs(y) * (1 - np.abs(y))).sum()
            return BaseDecisionTree.Node(node_type="LEAF", target_value=target_value)

        feature_num, threshold = self.find_best_split(X, y)
        left_idx = np.argwhere(X[:, feature_num] <= threshold).reshape(-1)
        right_idx = np.setdiff1d(np.arange(X.shape[0]), left_idx).reshape(-1)
        left_node = self.create_node(X[left_idx], y[left_idx], depth + 1)
        right_node = self.create_node(X[right_idx], y[right_idx], depth + 1)
        return BaseDecisionTree.Node(feature_num, threshold, left_node, right_node)

    # TODO Add parallization by features
    def find_best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Find best split parametr e.g. feature and predicate value

        Parametrs
        ---------
        X : nd.array of shape [n_samples, n_features] 
            Array of samples in current split.
        y : nd.arrat of shape [n_samples] 
            Array of sample labels in current split.
        
        Return
        ---------
        feature_num, predicate_value : int, float
        """
        y = y.reshape(-1, 1)
        n_samples = X.shape[0]

        # multi processing library
        if self.criterion == "mse":
            X_sort_idx = X.argsort(axis=0)
            X_sorted = np.take_along_axis(X, X_sort_idx, axis=0)
            y_sorted = np.take_along_axis(np.tile(y, self.n_features), X_sort_idx, axis=0)

            pre_scores = np.cumsum(y_sorted, axis=0)

            samples_range = np.arange(1, n_samples)
            
            sub_path_coef = 1 / samples_range
            score =  sub_path_coef.reshape(-1, 1) * np.power(pre_scores[:-1], 2)

            super_path_coef = 1 / (n_samples - samples_range)
            score += super_path_coef.reshape(-1, 1) * np.power(pre_scores[-1] - pre_scores[:-1], 2)

            score[X_sorted[:-1] == X_sorted[1:]] = -1
            shift = self.min_samples_leaf - 1
            if shift:
                for x in score.T:
                    idx = np.argwhere(x >= 0)
                    idx = np.setdiff1d(np.arange(idx.size), idx[shift:-shift])
                    x[idx] = -1

            eps = 1e-10
            # Perfomance ??
            idxs = np.argwhere(abs(score - score.max()) < eps)
            x, y = idxs[np.random.choice(idxs.shape[0])]
            return y, X_sorted[x, y]
        else:
            raise Exception("Criterion {} is not realized yet".format(self.criterion))
