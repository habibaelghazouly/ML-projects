import torch

class LogisticRegressionScratch:
    def __init__(self, input_dim, learning_rate=0.01, max_epochs=30):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.W = torch.zeros((input_dim, 1), dtype=torch.float32)
        self.b = torch.zeros(1, dtype=torch.float32)

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def binary_cross_entropy(self, y_pred, y_true):
        eps = 1e-8
        return -torch.mean(y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps))

    def forward_pass(self, X_batch):
        scores = X_batch @ self.W + self.b
        y_pred = self.sigmoid(scores)
        return y_pred

    def compute_accuracy(self, y_pred, y_true):
        preds = (y_pred >= 0.5).float()
        return (preds.squeeze() == y_true).float().mean().item()

    def compute_gradients(self, X_batch, y_batch, y_pred):
        error = y_pred - y_batch.unsqueeze(1)
        dw = (X_batch.T @ error) / X_batch.shape[0]
        db = error.mean()
        return dw, db

    def update_weights(self, dw, db):
        self.W -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def fit(self, train_loader, val_loader):
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(self.max_epochs):
            train_loss, train_acc = 0, 0
            for X_batch, y_batch in train_loader:
                y_pred = self.forward_pass(X_batch)
                loss = self.binary_cross_entropy(y_pred, y_batch)
                dw, db = self.compute_gradients(X_batch, y_batch, y_pred)
                self.update_weights(dw, db)
                train_loss += loss.item()
                train_acc += self.compute_accuracy(y_pred, y_batch)
            train_losses.append(train_loss / len(train_loader))
            train_accs.append(train_acc / len(train_loader))

            # Validation
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = self.forward_pass(X_batch)
                    loss = self.binary_cross_entropy(y_pred, y_batch)
                    val_loss += loss.item()
                    val_acc += self.compute_accuracy(y_pred, y_batch)
            val_losses.append(val_loss / len(val_loader))
            val_accs.append(val_acc / len(val_loader))

            print(f"Epoch [{epoch+1}/{self.max_epochs}] "
                  f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
                  f"Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

        return train_losses, val_losses, train_accs, val_accs

    def predict(self, X):
        with torch.no_grad():
            y_pred = self.forward_pass(X)
            return (y_pred >= 0.5).float().squeeze()
