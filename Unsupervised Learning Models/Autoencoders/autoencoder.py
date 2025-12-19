import numpy as np
from activations import activations
from loss import compute_mse_loss

# layers [input => h1 => ... => bottleneck => ... h1 => output]
# def __init__(self , input_dim ,  hidden_layers[] , bottleneck , activation = 'relu' , lr = 0.01 , lamda = 0.1):
class Autoencoder:
    def __init__(self , layers_dims, activation = 'relu' , lr = 0.01 , lamda = 0.1):
        self.layers = layers_dims   
        self.activation = activation
        self.lr = lr        
        self.lamda = lamda
        self.L = len(layers_dims) - 1  # Number of layers (arrows) , array feha nodes : number of neurons fe kol layer
       
        # initialize weights and biases
        self.weights = [np.random.randn(layers_dims[i+1], layers_dims[i]) * 0.01 for i in range(self.L)]
        self.biases = [np.zeros((layers_dims[i+1], 1)) for i in range(self.L)]
    
    
    def activate(self, z):
        if self.activation == 'relu':
            return activations.relu(z)
        elif self.activation == 'sigmoid':
            return activations.sigmoid(z)
        elif self.activation == 'tanh':
            return activations .tanh(z)
        else:
            raise ValueError("Unsupported activation function")
    
    def deriv_activate(self, z):
        if self.activation == 'relu':
            return activations.relu_deriv(z)
        elif self.activation == 'sigmoid':
            return activations.sigmoid_deriv(z)
        elif self.activation == 'tanh':
            return activations.tanh_deriv(z)
        else:
            raise ValueError("Unsupported activation function")
        
    def lr_schedule(self, epoch, initial_lr, decay_rate):
        return initial_lr / (1 + decay_rate * epoch)    
    
    def forward(self, x):
        self.a = []
        self.h = [x]

        for l in range(self.L):
            a = self.weights[l] @ self.h[-1] + self.biases[l]
            self.a.append(a)

            if l == self.L - 1:
                self.h.append(a)  # linear output
            else:
                self.h.append(self.activate(a))

        return self.h[-1]
    
    def backward(self, x):
        m = x.shape[1]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # Compute output layer error
        delta = self.h[-1] - x

        for l in reversed(range(self.L)):
            grads_w[l] = (delta @ self.h[l].T) / m
            grads_b[l] = np.sum(delta, axis=1, keepdims=True) 

            if l > 0:
                delta = (self.weights[l].T @ delta) * self.deriv_activate(self.a[l-1])
                grads_w[l] += (self.lamda) * self.weights[l]  # L2 regularization

        # Update weights and biases
        for l in range(self.L):
            self.weights[l] -= self.lr * grads_w[l]
            self.biases[l] -= self.lr * grads_b[l]

        return grads_w , grads_b
        
    def train(self, x , epochs, batch_size):
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[1])
            x_shuffled = x[:, permutation]

            epoch_loss = 0

            for i in range(0, x.shape[1], batch_size):
                x_batch = x_shuffled[:, i:i+batch_size]

                # Forward pass
                x_hat = self.forward(x_batch)  
                epoch_loss += compute_mse_loss(x_batch, x_hat) 

                # Backward pass
                grads_w, grads_b = self.backward(x_batch)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}") 