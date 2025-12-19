import numpy as np

# ---------------- Activation functions ----------------
class Activation:
    def forward(self, x): raise NotImplementedError
    def backward(self, dout): raise NotImplementedError

class ReLU(Activation):
    def __init__(self): self.mask = None
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    def backward(self, dout):
        return dout * self.mask

class Sigmoid(Activation):
    def __init__(self): self.out = None
    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out
    def backward(self, dout):
        return dout * (self.out * (1 - self.out))

class Tanh(Activation):
    def __init__(self): self.out = None
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    def backward(self, dout):
        return dout * (1 - self.out**2)

def get_activation(name: str):
    name = name.lower()
    if name == "relu": return ReLU()
    if name == "sigmoid": return Sigmoid()
    if name == "tanh": return Tanh()
    raise ValueError(f"Unknown activation: {name}")

# ---------------- Dense Layer ----------------
class Dense:
    def __init__(self, in_dim, out_dim, weight_scale=0.01):
        self.W = weight_scale * np.random.randn(in_dim, out_dim)
        self.b = np.zeros(out_dim)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        # gradients
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx

# ---------------- Autoencoder Model ----------------
class Autoencoder:
    """Fully-connected autoencoder with backprop + mini-batch SGD.

    Architecture:
      Encoder:  input -> h1 -> h2 -> h3 -> bottleneck
      Decoder:  bottleneck -> h3 -> h2 -> h1 -> output

    Hidden layers count: >= 3 in encoder and decoder (satisfied by default).
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        hidden_dims=(64, 32, 16),
        activation: str = "relu",
        lr: float = 1e-3,
        l2: float = 0.0,
        lr_decay: float = 1.0,
        min_lr: float = 1e-6,
        seed: int = 42
    ):
        np.random.seed(seed)
        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        if len(self.hidden_dims) < 3:
            raise ValueError("Need at least 3 hidden layers in encoder/decoder. Provide hidden_dims length >= 3.")
        self.activation_name = activation
        self.lr = float(lr)
        self.l2 = float(l2)
        self.lr_decay = float(lr_decay)
        self.min_lr = float(min_lr)

        act = lambda: get_activation(self.activation_name)

        # Encoder layers
        dims_enc = (self.input_dim,) + self.hidden_dims + (self.bottleneck_dim,)
        self.enc_layers = []
        self.enc_acts = []
        for i in range(len(dims_enc) - 1):
            self.enc_layers.append(Dense(dims_enc[i], dims_enc[i+1], weight_scale=np.sqrt(2/dims_enc[i])))
            if i != len(dims_enc) - 2:  # no activation after bottleneck by default
                self.enc_acts.append(act())
            else:
                self.enc_acts.append(None)

        # Decoder layers (mirror)
        dims_dec = (self.bottleneck_dim,) + self.hidden_dims[::-1] + (self.input_dim,)
        self.dec_layers = []
        self.dec_acts = []
        for i in range(len(dims_dec) - 1):
            self.dec_layers.append(Dense(dims_dec[i], dims_dec[i+1], weight_scale=np.sqrt(2/dims_dec[i])))
            if i != len(dims_dec) - 2:  # last is linear output
                self.dec_acts.append(act())
            else:
                self.dec_acts.append(None)

        self.loss_history = []
        self.lr_history = []

    # ----- forward passes -----
    def encode(self, X):
        out = X
        for layer, act in zip(self.enc_layers, self.enc_acts):
            out = layer.forward(out)
            if act is not None:
                out = act.forward(out)
        return out

    def decode(self, Z):
        out = Z
        for layer, act in zip(self.dec_layers, self.dec_acts):
            out = layer.forward(out)
            if act is not None:
                out = act.forward(out)
        return out

    def forward(self, X):
        Z = self.encode(X)
        Xhat = self.decode(Z)
        return Z, Xhat

    # ----- loss -----
    def mse_loss(self, X, Xhat):
        return np.mean((X - Xhat) ** 2)

    def l2_penalty(self):
        if self.l2 <= 0:
            return 0.0
        s = 0.0
        for layer in self.enc_layers + self.dec_layers:
            s += np.sum(layer.W ** 2)
        return 0.5 * self.l2 * s

    def loss(self, X, Xhat):
        return self.mse_loss(X, Xhat) + self.l2_penalty()

    # ----- backward -----
    def backward(self, X, Xhat):
        # d/dXhat of MSE: 2/N * (Xhat - X)
        N = X.shape[0]
        dout = (2.0 / (N * X.shape[1])) * (Xhat - X)

        # decoder backprop
        for layer, act in reversed(list(zip(self.dec_layers, self.dec_acts))):
            if act is not None:
                dout = act.backward(dout)
            dout = layer.backward(dout)

        # encoder backprop
        for layer, act in reversed(list(zip(self.enc_layers, self.enc_acts))):
            if act is not None:
                dout = act.backward(dout)
            dout = layer.backward(dout)

        # add L2 gradients
        if self.l2 > 0:
            for layer in self.enc_layers + self.dec_layers:
                layer.dW += self.l2 * layer.W

    def step(self):
        lr = self.lr
        for layer in self.enc_layers + self.dec_layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    def fit(self, X, epochs=200, batch_size=64, verbose=1, shuffle=True):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        for epoch in range(1, int(epochs) + 1):
            if shuffle:
                idx = np.random.permutation(n)
                Xb = X[idx]
            else:
                Xb = X

            epoch_losses = []
            for i in range(0, n, int(batch_size)):
                xb = Xb[i:i+int(batch_size)]
                _, xhat = self.forward(xb)
                L = self.loss(xb, xhat)
                epoch_losses.append(L)
                self.backward(xb, xhat)
                self.step()

            mean_loss = float(np.mean(epoch_losses))
            self.loss_history.append(mean_loss)
            self.lr_history.append(self.lr)

            # lr scheduling
            self.lr = max(self.min_lr, self.lr * self.lr_decay)

            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"Epoch {epoch:4d}/{epochs} | loss={mean_loss:.6f} | lr={self.lr:.6g}")

        return self

    def transform(self, X):
        return self.encode(np.asarray(X, dtype=float))

    def inverse_transform(self, Z):
        return self.decode(np.asarray(Z, dtype=float))

    def reconstruction_error(self, X):
        X = np.asarray(X, dtype=float)
        _, Xhat = self.forward(X)
        return float(np.mean((X - Xhat) ** 2))
