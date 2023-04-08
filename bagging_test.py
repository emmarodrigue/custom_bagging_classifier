import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import unittest
from bagging import CustomBaggingClassifier

class TestCustomBaggingClassifier(unittest.TestCase):
    
    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.clf = CustomBaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=10)
        self.clf.fit(self.X, self.y)

    def test_fit(self):
        # Check that the fitted model has the same number of estimators as specified in the constructor
        self.assertEqual(len(self.clf.trained_bootstrap_models_), self.clf.n_estimators_)
        # Check that the trained models are of the same type as the base estimator
        self.assertIsInstance(self.clf.trained_bootstrap_models_[0], DecisionTreeClassifier)
        # Check that the probabilities sum up to 1 for each instance
        probs = self.clf.predict_proba(self.X)
        self.assertTrue(np.allclose(probs.sum(axis=1), np.ones(self.X.shape[0])))
    
    def test_predict(self):
        # Check that the predicted classes are of the correct shape
        y_pred = self.clf.predict(self.X)
        self.assertEqual(y_pred.shape, self.y.shape)
        # Check that the predicted classes are integers and within the range of the original classes
        self.assertLessEqual(y_pred.min(), self.y.max())
        self.assertGreaterEqual(y_pred.max(), self.y.min())

    def test_predict_proba(self):
        # Check that the predicted probabilities are of the correct shape
        probs = self.clf.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.X.shape[0], len(self.clf.classes_)))
        # Check that the predicted probabilities are between 0 and 1
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))

    def test_get_bootstrap_sample(self):
        # Check that the bootstrap sample is of the correct shape
        bootstrap_sample_X, bootstrap_sample_y, oob_sample_X, oob_sample_y = self.clf._get_bootstrap_sample(self.X, self.y)
        self.assertEqual(bootstrap_sample_X.shape[0], bootstrap_sample_y.shape[0])
        self.assertEqual(oob_sample_X.shape[0], oob_sample_y.shape[0])
        # Check that the bootstrap sample is a subset of the original data
        self.assertTrue(np.all(np.isin(bootstrap_sample_X, self.X)))
        self.assertTrue(np.all(np.isin(bootstrap_sample_y, self.y)))
        # Check that the OOB sample is a subset of the original data
        self.assertTrue(np.all(np.isin(oob_sample_X, self.X)))
        self.assertTrue(np.all(np.isin(oob_sample_y, self.y)))
