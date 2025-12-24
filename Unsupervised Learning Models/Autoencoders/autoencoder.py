from mimetypes import init
import numpy as np
import torch
from .activations import activations
from .loss import compute_mse_loss
import torch.nn.init as init

class Autoencoder:
    # layers [input => h1 => ... => bottleneck => ... h1 => output]
    # def __init__(self , layers_dims, activation = 'relu' , lr = 0.01 , lamda = 0.1):
    #     self.layers = layers_dims   
    #     self.activation = activation
    #     self.lr = lr        
    #     self.lamda = lamda
    #     self.L = len(layers_dims) - 1  # Number of layers (arrows) , array feha nodes : number of neurons fe kol layer
       
    def __init__(self , input_dim , hidden_layers , bottleneck , activation='tanh' , lr=0.01 , lamda=0.1, seed=42):
        
        self.lr = lr
        self.lamda = lamda
        self.activation = activation
        np.random.seed(seed)
        torch.manual_seed(seed)

        # build encoder + bottleneck + decoder (inverse)
        encoder_layers = hidden_layers
        decoder_layers = hidden_layers[::-1]
        layer_dims = [input_dim] + encoder_layers + [bottleneck] + decoder_layers + [input_dim]

        self.num_layers = len(layer_dims) - 1 # number of edges
        self.weights = []
        self.biases = []

        # initialize random weights and zero biases
        
        self.weights = []
        for i in range(self.num_layers):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i+1]

            # create torch weight tensor
            w = torch.empty(fan_out, fan_in)
            init.xavier_uniform_(w)

            # convert back to numpy
            self.weights.append(w.detach().cpu().numpy())

        self.biases = [np.zeros((layer_dims[i+1], 1)) for i in range(self.num_layers)]
    
    
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

    # step decay learning rate    
    def lr_schedule(self, epoch, initial_lr, decay_rate = 0.01, step_size=10):
        return initial_lr / (1 + decay_rate * (epoch // step_size))
  
    
    def forward(self, x):
        self.a = [] 
        self.h = [x] # activations


        for l in range(self.num_layers):
            a = self.weights[l] @ self.h[-1] + self.biases[l]
            self.a.append(a)

            if l == self.num_layers - 1:
                self.h.append(a)  # linear output
            else:
                self.h.append(self.activate(a))

        # returns the reconstructed output
        return self.h[-1]
    
    def backward(self, x):
        m = x.shape[1]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # Compute output layer error
        delta = self.h[-1] - x

        for l in reversed(range(self.num_layers)):
            grads_w[l] = (delta @ self.h[l].T) / m
            grads_b[l] = np.sum(delta, axis=1, keepdims=True)/m

            if l > 0:
                delta = (self.weights[l].T @ delta) * self.deriv_activate(self.a[l-1])
                #grads_w[l] += (self.lamda) * self.weights[l]  # L2 regularization

        # Update weights and biases
        for l in range(self.num_layers):
            self.weights[l] -= self.lr * grads_w[l]
            self.biases[l] -= self.lr * grads_b[l]

        return grads_w , grads_b
        
    def train(self, x , epochs, batch_size , schedule=True , decay_rate=0.01):
        n = x.shape[0]
        x = x.T  # Transpose for easier batch processing
        losses = []

        for epoch in range(epochs):
            permutation = np.random.permutation(n)
            x_shuffled = x[:, permutation]
            
            epoch_loss = 0

            for i in range(0, n, batch_size):
                x_batch = x_shuffled[:, i:i+batch_size]

                # Forward pass
                x_hat = self.forward(x_batch)  
                epoch_loss += compute_mse_loss(x_batch, x_hat) 

                # Backward pass
                grads_w, grads_b = self.backward(x_batch)

            losses.append(epoch_loss)

            if schedule:
                self.lr = self.lr_schedule(epoch, self.lr, decay_rate , step_size=10)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}") 
            print(f"Learning Rate: {self.lr:.6f}")
            print("-" * 30)
        return losses
   
    def predict(self, x):
        x = np.asarray(x)
        # ensure x_in has shape (features, samples)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[0] == self.weights[0].shape[1]:
            x_in = x  # already (features, samples)
        else:
            x_in = x.T  # convert (samples, features) -> (features, samples)
        x_hat = self.forward(x_in)
        return x_hat.T  # return (samples, features)