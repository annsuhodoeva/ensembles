import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
import random
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.tree = [DecisionTreeRegressor(max_depth=max_depth, **trees_parameters) for _ in range(n_estimators)]
        self.feature_subsample_size = feature_subsample_size
        
    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        self.feat = []
        for i in range(self.n_estimators):
            n = X.shape[0] // self.n_estimators
            obj = random.sample(list(range(X.shape[0])), n)
            if self.feature_subsample_size is None:
                self.feat.append(list(range(X.shape[1])))
            else:
                self.feat.append(random.sample(list(range(X.shape[1])), self.feature_subsample_size))
            self.tree[i].fit(X[obj, :][:, self.feat[i]], y[obj])

        
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        s = 0
        for i in range(self.n_estimators):
            s += self.tree[i].predict(X[:, self.feat[i]])
        return s / self.n_estimators


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            Use learning_rate * gamma instead of gamma

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.learning_rate = learning_rate
        self.trees_parameters = trees_parameters
        
        
    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        self.tree = []
        self.feat = []
        self.c = []
        w = 0
        n = X.shape[0] // self.n_estimators
        for i in range(self.n_estimators):
            t = DecisionTreeRegressor(max_depth=self.max_depth, **self.trees_parameters)
            obj = random.sample(list(range(X.shape[0])), n)
            if self.feature_subsample_size is None:
                self.feat.append(list(range(X.shape[1])))
            else: 
                self.feat.append(random.sample(list(range(X.shape[1])), self.feature_subsample_size))
            t.fit(X[obj, :][:, self.feat[i]], (y - w)[obj])
            self.c.append(minimize_scalar(lambda x: mean_squared_error(w + x * t.predict(X), y)).x)
            self.tree.append(t)
            w += self.learning_rate * self.c[i] * t.predict(X)

    def predict(self, X):
        pr = 0
        for i in range(self.n_estimators):
            pr += self.learning_rate * self.c[i] * self.tree[i].predict(X[:, self.feat[i]])
        return pr
        
