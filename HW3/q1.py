import numpy as np
import sklearn.linear_model as skl

class FeatureSelection:
   def rank_correlation(self, x, y):
       ranks = np.zeros(y.shape[0])
       #your code here
       return ranks

   def lasso(self, x, y):
       ranks = np.zeros(y.shape[0])
       # your code here
       return ranks

    def stepwise(self, x, y):
        ranks = np.zeros(y.shape[0])
        # your code here
        return ranks

class Regression:
    def ridge_lr(self, train_x, train_y, test_x, test_y):
        test_prob = np.zeros(test_x.shape[0])
        return test_prob

    def tree_regression(self, train_x, train_y, test_x, test_y):
        test_prob = np.zeros(test_x.shape[0])
        return test_prob

