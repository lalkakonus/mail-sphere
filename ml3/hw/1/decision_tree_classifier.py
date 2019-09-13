import numpy as np
from sklearn.preprocessing import normalize

class Node():
    def __init__(self, feature_num=None, predicate_value=None, left_node=None, right_node=None, node_type="NONLEAF",
                 class_probabilities=None):
        self.feature_num = feature_num
        self.predicate_value = predicate_value
        self.right_node = right_node
        self.left_node = left_node
        self.class_probabilities = class_probabilities
        self.type = node_type

    def predict(self, X):
        if self.type = "LEAF":
            return self.class_probabilities
        else:
            if X[self.feature_num] < self.predicate_value:
                return self.left.predict(X)
            else:
                return self.rigth.predict(X)

    @property
    def right(self):
        return self.right

    @property
    def left(self):
        return self.left



class DecisionTreeClassifier():

    def __inti__(self,criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize Decision tree classifier

        Parameters
        ---------
        criterion: string, optional (deffailt="gini")
            The function to measure the quality of a split. Supported criteria
            are "gini" for the Gini imputity, which is equal to 1 - sum_{i=1}^{N}(p_i)
        max_deptx: maximum avaliable depth
        min_samples_split: minimum count of samples in split to split it into
        min_samples_leaf:  minimim cout of samples in split to make leaf node
        """
        self.criterion=criterion
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.root = None
        
    def create_node(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Find best split parametr e.g. feature and predicate value and create NONLEAF node
        in case it is possible, ether create LEAF node with classes probabilities

        Parametrs
        ---------
        X : nd.array of shape [n_samples, n_features] - array of samples in current split
        y : nd.arrat of shape [n_samples] - array of samples label in current split
        depth : int >= 0 - current node depth
        
        Return
        ---------
        node : Node - LEAF (with class probabilities attributes) 
               or NONLEAF (with feature and predicate value attributes)
        """
        if (depth >= self.max_depth or 
            y.size < max(min_samples_leaf * 2, min_samples_split * 2)):
            # if split is too small or branch deep enought, create leaf node
            proba = np.zeros(self.n_classes)
            classes_cnt = np.unique(y, return_counts=True)
            proba[classes_cnt[0]] = classes_cnt[1]
            return Node(node_type="LEAF", class_probabilities=normalize(proba))


        feature_num, predicate_value = self.find_best_split(X, y)
        left_samples = np.argwhere(X[:, feature_num] < predicate_value)
        right_samples = np.setdiff1d(np.arange(n_feature), left_samples)
        left_node = self.create_node(X[left], y[left], depth + 1)
        right_node = self.create_node(X[right], y[right], depth + 1)
        return Node(feature_num, predicate_value, left_node, right_node)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train model using X 
        
        Parametrs
        ---------
        X : nd.array of shape [n_samples, n_features] - array of samples in current split
        y : nd.arrat of shape [n_samples] - array of samples label in current split
        depth : int >= 0 - current node depth
        
        Return
        ---------
        self : object 
        """
        self.n_classes = np.unique(y)
        self.n_features = X.shape(1)
        self.root = self.create_node(X, y, depth=0)
        return self

    def predict(self, X):
        return np.array([self.root.predict(sample) for sample in X]).reshape=(-1, 1)


    def score(self, X, y):
        y_predicted = self.predict(X)
        return accuracy_score(y_true=y, y_predicted=y_predicted)

    def find_best_split(self, X: np.ndarray, y: np.ndarray):
        # Add parallization
        # Test function

        if self.criterion == "gini":
            n_samples = X.shape[0]
            X_argsorted = np.argsort(X, axis=0)
            X_sorted = np.take_along_axis(X, X_argosrted, axis=0)
            y_sorted = np.take_along_axis(np.tile(y, self.n_features), X_argsorted, axis=0)

            pre_scores = np.zeros(list(X.shape) + [self.n_classes])
            for pos, classes in enumerate(y_sorted, 1):
                pre_scores[pos] = pre_scores[pos - 1]
                pre_scores[:, range(pre_scores.shape[1]), y_sorted[:, pos]] += 1

            pre_scores = np.power(pre_scores, 2).sum(axis=2)

            samples_range = np.arange(n_samples)
            gini = pre_scores(n_samples  - 2 * samples_range) / (samples_range * (n_samples - samples_range))
            gini = pre_scores * gini.reshape(-1, 1)

            gini += (n_samples - sample_range).reshape(-1, 1) * pre_scores[-1].reshape(1, -1)
            x, y =  gini.argmax() // gini.shape[1], gini.argmax() % gini.shape[1]
            return x, X_sorted[x, y]
        else:
            raise Exception("Criterion {} is not realized yet".format(self.criterion))
