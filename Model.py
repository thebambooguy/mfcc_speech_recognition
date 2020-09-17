from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from abc import ABC


class Model(ABC):
    def __init__(self, X_train=None, y_train=None, model_filename=None):
        self.X_train = X_train
        self.y_train = y_train
        self.model_filename = model_filename

    def fit(self, X, y):
        self.internal_model.fit(X, y)

    def predict(self, X):
        temp = self.internal_model.predict(X)
        return temp

    def save_model(self):
        dump(self, self.model_filename)

    def load_model(self):
        self.model = load(self.model_filename)
        return self

    def gridsearchCV(self):
        best_model = GridSearchCV(self.internal_model, self.parameters, cv=self.cv, scoring=self.scoring)
        return best_model

    def set_internal_model(self, external_model):
        self.internal_model = external_model


class LogisticRegressionModel(Model):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1,
                 class_weight=None, random_state=None, max_iter=100, solver='newton-cg'):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.solver = solver
        self.internal_model = LogisticRegression(penalty=self.penalty, dual=self.dual, tol=self.tol, C=self.C,
                                                 fit_intercept=self.fit_intercept,
                                                 intercept_scaling=self.intercept_scaling,
                                                 random_state=self.random_state, max_iter=self.max_iter)
        self.model_filename = "LogisticRegressionModelparameters.joblib"

        self.parameters = {'penalty': ['l2'], 'dual': [True, False], 'C': [1.0, 3.0, 5.0, 15.0],
                           'fit_intercept': [True, False], 'max_iter': [100, 500, 1000]}
        self.cv = 2
        self.scoring = 'balanced_accuracy'


class RandomForestModel(Model):
    def __init__(self, n_estimators=5, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="auto",
                 max_leaf_nodes=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.internal_model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                     min_samples_split=self.min_samples_split,
                                                     min_samples_leaf=self.min_samples_leaf,
                                                     max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes,
                                                     random_state=self.random_state)
        self.model_filename = "RandomForestModelparameters.joblib"

        self.parameters = {'n_estimators': [10, 50, 100, 150, 200, 15], 'max_depth': [None, 2, 5, 10, 6],
                           'max_leaf_nodes': [None, 2, 5]}
        self.cv = 2
        self.scoring = 'balanced_accuracy'