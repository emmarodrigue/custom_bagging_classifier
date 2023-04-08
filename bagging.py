from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import numpy as np
import random

class CustomBaggingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, random_state=None):
        """
        Parameters
        ----------
        base_estimator : object or None, optional (default=None)    The base estimator to fit on random subsets of the dataset. 
                                                                    If None, then the base estimator is a decision tree.
        n_estimators : int, optional (default=10)                   The number of base estimators in the ensemble.
        random_state : int or None, optional (default=None)         Controls the randomness of the estimator. 
        """
        self.base_estimator = base_estimator
        self.n_estimators_ = n_estimators
        self.random_state = random_state
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(criterion='entropy', random_state=self.random_state, max_depth=2)
        if self.random_state is None:
            self.random_state = 43

    def fit(self, X, y):
        """
        Build a Bagging classifier from the training set (X, y).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)                 The input samples.
        y : ndarray of shape (n_samples,)                            The target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.estimators_ = []
        self.weights_ = []
        self.trained_bootstrap_models_ = []

        random.seed(self.random_state)

        for _ in range(self.n_estimators_):
            estimator = clone(self.base_estimator)
            bootstrap_X, bootstrap_y, oob_X, oob_y = self._get_bootstrap_sample(X, y)
            trained_model = estimator.fit(bootstrap_X, bootstrap_y)
            oob_accuracy = trained_model.score(oob_X, oob_y)
            self.weights_.append(oob_accuracy)
            self.estimators_.append(estimator)
            self.trained_bootstrap_models_.append(trained_model)

        self.weights_ = np.array(self.weights_) / np.sum(self.weights_)

        return self

    def predict(self, X):
        """
        Predict class for X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)                 The input samples.
        
        Returns
        -------
        pred : ndarray of shape (n_samples,)                         The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        probas = self.predict_proba(X)
        pred = np.argmax(probas, axis=1)
        return self.classes_[pred]

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)                 The input samples.

        Returns
        -------
        probas : ndarray of shape (n_samples, n_classes)             The class probabilities of the input samples. The order of 
                                                                     the classes corresponds to that in the attribute classes_.
        """

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        probas = np.zeros((X.shape[0], len(self.classes_)))
        for trained_model, weight in zip(self.trained_bootstrap_models_, self.weights_):
            probas += weight * trained_model.predict_proba(X)

        return probas

    def _get_bootstrap_sample(self, X, y):
        """
        Returns a bootstrap sample of the same size as the original input X, 
        and the out-of-bag (oob) sample. According to the theoretical analysis, about 63.2% 
        of the original indexes will be included in the bootsrap sample. Some indexes will
        appear multiple times.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)                  The input samples.
        y : ndarray of shape (n_samples,)                             The target values.

        Returns
        -------
        bootstrap_sample_X : ndarray of shape (n_samples, n_features) The bootstrap sample of the input samples.
        bootstrap_sample_y : ndarray of shape (n_samples,)            The bootstrap sample of the target values.
        oob_sample_X : ndarray of shape (n_samples, n_features)       The out-of-bag sample of the input samples.
        oob_sample_y : ndarray of shape (n_samples,)                  The out-of-bag sample of the target values.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        n_samples = X.shape[0]
        bootstrap = np.random.choice(np.arange(n_samples), size=n_samples)
        oob = np.array([i for i in range(n_samples) if i not in bootstrap])

        bootstrap_sample_X = X[bootstrap]
        bootstrap_sample_y = y[bootstrap]
        oob_sample_X = X[oob]
        oob_sample_y = y[oob]

        return bootstrap_sample_X, bootstrap_sample_y, oob_sample_X, oob_sample_y

