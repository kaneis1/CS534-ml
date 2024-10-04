import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
import time

test_path = 'D:\\code\\latex\\CS534-ml\\HW2\\data\\spam.test.dat'
train_path = 'D:\\code\\latex\\CS534-ml\\HW2\\data\\spam.train.dat'

# Load your data
train_data = pd.read_csv(train_path, delimiter=' ', header=None, dtype=np.float64).to_numpy()
test_data = pd.read_csv(test_path, delimiter=' ', header=None, dtype=np.float64).to_numpy()
trainx, trainy = train_data[:, :-1], train_data[:, -1]
testx, testy = test_data[:, :-1], test_data[:, -1]

# Best parameters from k-fold and MCCV
best_params_kfold_ridge = {'alpha': 65.79332246575683}
best_params_kfold_lasso = {'alpha': 0.0013530477745798076}
best_params_mccv_ridge = {'valsize': 0.1, 's': 5, 'alpha': 0.14174741629268062}
best_params_mccv_lasso = {'valsize': 0.1, 's': 5, 'alpha': 0.0005336699231206312}

def train_and_evaluate(model, trainx, trainy, testx, testy):
    start_time = time.time()
    model.fit(trainx, trainy)
    train_time = time.time() - start_time
    
    test_preds = model.predict(testx)
    
    test_mse = mean_squared_error(testy, test_preds)
    test_acc = accuracy_score(testy, np.round(test_preds))
    test_auc = roc_auc_score(testy, test_preds)
    
    return test_acc, test_mse, test_auc, train_time

# Train and evaluate models
ridge_kfold = Ridge(alpha=best_params_kfold_ridge['alpha'])
lasso_kfold = Lasso(alpha=best_params_kfold_lasso['alpha'])
ridge_mccv = Ridge(alpha=best_params_mccv_ridge['alpha'])
lasso_mccv = Lasso(alpha=best_params_mccv_lasso['alpha'])

results = []

# Ridge (k-fold)
test_acc, test_mse, test_auc, train_time = train_and_evaluate(ridge_kfold, trainx, trainy, testx, testy)
results.append(['Ridge (k-fold)', test_acc, test_mse, test_auc, train_time])

# Lasso (k-fold)
test_acc, test_mse, test_auc, train_time = train_and_evaluate(lasso_kfold, trainx, trainy, testx, testy)
results.append(['Lasso (k-fold)', test_acc, test_mse, test_auc, train_time])

# Ridge (MCCV)
test_acc, test_mse, test_auc, train_time = train_and_evaluate(ridge_mccv, trainx, trainy, testx, testy)
results.append(['Ridge (MCCV)', test_acc, test_mse, test_auc, train_time])

# Lasso (MCCV)
test_acc, test_mse, test_auc, train_time = train_and_evaluate(lasso_mccv, trainx, trainy, testx, testy)
results.append(['Lasso (MCCV)', test_acc, test_mse, test_auc, train_time])

# Print results
results_df = pd.DataFrame(results, columns=['Model', 'Test Accuracy', 'Test MSE', 'Test AUC', 'Training Time'])
print(results_df.to_string(index=False))