import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Define the objective function f_o(x)
def objective_function(X, Y, beta, lambd, alpha):
    l2_reg = alpha * np.sum(beta ** 2)
    l1_reg = (1 - alpha) * np.sum(np.abs(beta))
    
    residual = Y - X @ beta
    ls_loss = 0.5 * np.sum(residual ** 2)
    
    return ls_loss + lambd * (l2_reg + l1_reg)

# Gradient step function
def grad_step(x, y, beta, lambd, alpha, eta):
    grad_g = x.T @ (x @ beta - y) + 2 * lambd * alpha * beta
    beta_new = beta - eta * grad_g
    
    # Proximal operator for L1 regularization
    for i in range(len(beta_new)):
        if beta_new[i] > lambd * (1 - alpha) * eta:
            beta_new[i] -= lambd * (1 - alpha) * eta
        elif beta_new[i] < -lambd * (1 - alpha) * eta:
            beta_new[i] += lambd * (1 - alpha) * eta
        else:
            beta_new[i] = 0
    
    return beta_new

# Train the model for different alpha values
def train_elastic_net(X_train, y_train, lambd, eta, alpha, epochs=100):
    beta = np.zeros(X_train.shape[1])  # Initialize beta
    
    for epoch in range(epochs):
        beta = grad_step(X_train, y_train, beta, lambd, alpha, eta)
    
    return beta

# RMSE calculation
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Elastic Net parameter exploration
def parameter_exploration(X_train, y_train, X_val, y_val, X_test, y_test, lambd=0.8, eta=0.25, epochs=100):
    alpha_values = np.arange(0, 1.1, 0.1)
    results = []

    for alpha in alpha_values:
        # Train the model
        beta = train_elastic_net(X_train, y_train, lambd, eta, alpha, epochs)
        
        # Predict on training, validation, and test sets
        y_train_pred = X_train @ beta
        y_val_pred = X_val @ beta
        y_test_pred = X_test @ beta
        
        # Calculate RMSE and R^2 for each set
        train_rmse = calculate_rmse(y_train, y_train_pred)
        val_rmse = calculate_rmse(y_val, y_val_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Append the results
        results.append({
            'alpha': alpha,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2
        })
    
    return results

# Print results in a table format
def print_results_table(results):
    print(f"{'Alpha':<8}{'Train RMSE':<12}{'Val RMSE':<12}{'Test RMSE':<12}{'Train R^2':<12}{'Val R^2':<12}{'Test R^2':<12}")
    print("-" * 72)
    for res in results:
        print(f"{res['alpha']:<8.2f}{res['train_rmse']:<12.4f}{res['val_rmse']:<12.4f}{res['test_rmse']:<12.4f}"
              f"{res['train_r2']:<12.4f}{res['val_r2']:<12.4f}{res['test_r2']:<12.4f}")

if __name__ == '__main__':
    # Simulate synthetic data or load actual data
    np.random.seed(0)
    
    # Synthetic Data Generation
    X_train = np.random.randn(100, 10)  # 100 samples, 10 features for training
    y_train = X_train @ np.random.randn(10) + 0.5 * np.random.randn(100)

    X_val = np.random.randn(50, 10)  # 50 samples for validation
    y_val = X_val @ np.random.randn(10) + 0.5 * np.random.randn(50)

    X_test = np.random.randn(50, 10)  # 50 samples for testing
    y_test = X_test @ np.random.randn(10) + 0.5 * np.random.randn(50)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Explore the parameter alpha
    results = parameter_exploration(X_train, y_train, X_val, y_val, X_test, y_test, lambd=0.8, eta=0.001, epochs=100)

    # Print the results
    print_results_table(results)
