import numpy as np
import q2
import matplotlib.pyplot as plt
train_path='HW1\energydata\energy_train.csv'
val_path='HW1\energydata\energy_val.csv'
test_path='HW1\energydata\energy_test.csv'

def loss(x, y, beta, el, alpha):
    l2_reg = (1 - alpha) * np.sum(np.abs(beta))
    l1_reg = alpha * np.sum(beta ** 2)
    
    ls_loss = 0.5 * np.sum((y - x @ beta) ** 2)
    
    return ls_loss + el * (l1_reg + l2_reg)

def grad_step(x, y, beta, el, alpha, eta):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    grad_g = x.T @ (x @ beta - y) + 2 * el * alpha * beta
    beta_new = beta - eta * grad_g
    for i in range(len(beta_new)):
        if beta_new[i] > el * (1 - alpha) * eta:
            beta_new[i] -= el * (1 - alpha) * eta
        elif beta_new[i] < -el * (1 - alpha) * eta:
            beta_new[i] += el * (1 - alpha) * eta
        else:
            beta_new[i] = 0
    
    return beta_new


class ElasticNet:
    def __init__(self, el, alpha, eta, batch, epoch):
       
        self.el = el
        self.alpha = alpha
        self.eta = eta
        self.batch = batch
        self.epoch = epoch
        self.beta = None
        return

    def coef(self):
        return self.beta

    def train(self, x, y):
        N, d = x.shape
        self.beta = np.zeros(d)  
        
        loss_history = {}
        
        
        for ep in range(self.epoch):
            
            indices = np.random.permutation(N)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            
            for i in range(0, N, self.batch):
                x_batch = x_shuffled[i:i+self.batch]
                y_batch = y_shuffled[i:i+self.batch]
                
                
                for xi, yi in zip(x_batch, y_batch):
                    self.beta = grad_step(xi, yi, self.beta, self.el, self.alpha, self.eta)
            
            
            epoch_loss = loss(x, y, self.beta, self.el, self.alpha)
            loss_history[ep+1]=epoch_loss

            
        return loss_history

    def predict(self, x):
        return x @ self.beta

if __name__ == '__main__':
    el_net = ElasticNet(el=6, alpha=0.5, eta=0.0001, batch=32, epoch=100)
    train_data,val_data,test_data=q2.load_data()
    trainx, trainy = q2.split_features_target(train_data)
    valx, valy = q2.split_features_target(val_data)
    testx, testy = q2.split_features_target(test_data)
    
    trainx,valx,testx=q2.preprocess_data(trainx,valx,testx)
    # # Train the model
    # history = el_net.train(trainx, trainy)
    # print('history:',history)
    # # Predict new values
    # predictions = el_net.predict(testx)
    # print('predictions:',predictions)
    # # Get the learned cofficients
    # coefficients = el_net.coef()
    # print('coefficients:',coefficients)
    
   
    np.random.seed(0)
    X = np.random.randn(100, 10)  # 100 samples, 10 features
    true_beta = np.random.randn(10)
    Y = X @ true_beta + 0.5 * np.random.randn(100)

    
    lambd = 0.8
    #alpha = 0.5
    epochs = 100
    eta=0.25
    initial_beta = np.zeros(10)


    # Store the loss history for each learning rate
    loss_histories = {}

    # Train the model with different learning rates
    for alpha in range(0,100,5):
        beta = initial_beta.copy()
        losses = []
        
        for epoch in range(epochs):
            beta = grad_step(X, Y, beta, lambd, alpha, eta)
            current_loss = loss(X, Y, beta, lambd, alpha)
            losses.append(current_loss)
        
        loss_histories[alpha] = losses

    # Plot the results
    plt.figure(figsize=(20, 10))
    for alpha, losses in loss_histories.items():
        plt.plot(range(60,epochs), losses[60:], label=f'alpha={alpha}')

    plt.title('Objective Value $f_o(x)$ for Different Learning Rates ($\lambda=0.8$, $\\eta=0.25$)')
    plt.xlabel('Epoch')
    plt.ylabel('Objective Value $f_o(x)$')
    plt.legend()
    plt.grid(True)
    plt.show()
    