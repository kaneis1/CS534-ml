import numpy as np
import sklearn.linear_model as skl
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

class FeatureSelection:
    def rank_correlation(self, x, y):
        correlations = [pearsonr(x[:, i], y)[0] for i in range(x.shape[1])]
        abs_correlations = np.abs(correlations)
        ranks = np.argsort(-abs_correlations)
        return ranks

    def lasso(self, x, y):
        ranks = np.zeros(y.shape[0])
        alpha = 0.1
        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(x, y)
        coefficients = lasso_model.coef_
        non_zero_indices = np.where(coefficients != 0)[0]
        non_zero_coefficients = coefficients[non_zero_indices]
        ranks = non_zero_indices[np.argsort(-np.abs(non_zero_coefficients))]
        return ranks

    def stepwise(self, x, y):
        ranks = np.zeros(y.shape[0])
        
        model = LinearRegression()

        sfs = SequentialFeatureSelector(
            model, 
            n_features_to_select="auto",  
            direction='forward',           
            scoring='neg_root_mean_squared_error',  
            cv=5,  
            n_jobs=-1 
        )
        
        sfs.fit(x, y)

        ranks = np.where(sfs.get_support())[0]

        return ranks

class Regression:
    def ridge_lr(self, train_x, train_y, test_x, test_y):
        test_prob = np.zeros(test_x.shape[0])
        model = Ridge(alpha=1.0)
        model.fit(train_x, train_y)
        test_prob = model.predict(test_x)
        return test_prob

    def tree_regression(self, train_x, train_y, test_x, test_y):
        test_prob = np.zeros(test_x.shape[0])
        model = DecisionTreeRegressor(max_depth=None, min_samples_leaf=1)
        model.fit(train_x, train_y)
        
        test_prob = model.predict(test_x)
        return test_prob

