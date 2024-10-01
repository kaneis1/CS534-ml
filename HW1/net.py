import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Function to return coefficients and performance metrics
def best_elastic_net_model(X_train, y_train, X_test, y_test, lambd=0.8, eta=0.001, epochs=100):
    alpha_values = np.arange(0, 1.1, 0.1)
    best_model = None
    best_rmse = float('inf')
    best_r2 = -float('inf')
    best_alpha = None
    best_beta = None
    
    # Train the Elastic Net for each alpha
    for alpha in alpha_values:
        beta = np.zeros(X_train.shape[1])  # Initialize beta
        
        # Gradient descent for each alpha value
        for epoch in range(epochs):
            beta = grad_step(X_train, y_train, beta, lambd, alpha, eta)
        
        # Predictions on the test set
        y_test_pred = X_test @ beta
        
        # Calculate RMSE and R^2
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Select the best model based on test RMSE
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_r2 = test_r2
            best_alpha = alpha
            best_beta = beta
    
    # Return best coefficients and performance metrics
    return {
        'best_alpha': best_alpha,
        'best_rmse': best_rmse,
        'best_r2': best_r2,
        'best_coefficients': best_beta
    }

# Example usage
if __name__ == '__main__':
    # Simulate some synthetic data or use actual data
    np.random.seed(0)
    X_train = np.random.randn(100, 10)  # 100 samples, 10 features
    y_train = X_train @ np.random.randn(10) + 0.5 * np.random.randn(100)

    X_test = np.random.randn(50, 10)  # 50 test samples
    y_test = X_test @ np.random.randn(10) + 0.5 * np.random.randn(50)
    
    # Feature scaling (important for elastic net)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Find the best Elastic Net model
    best_model_results = best_elastic_net_model(X_train, y_train, X_test, y_test, lambd=0.8, eta=0.001, epochs=100)
    
    print("Best Alpha:", best_model_results['best_alpha'])
    print("Best Test RMSE:", best_model_results['best_rmse'])
    print("Best Test R^2:", best_model_results['best_r2'])
    print("Best Coefficients:", best_model_results['best_coefficients'])
