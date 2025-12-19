import numpy as np  

def compute_mse_loss(x , x_hat):

    return np.mean(np.sum((x - x_hat) ** 2 , axis = 0)) /2 # mean over features