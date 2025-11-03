import torch
import torch.nn.functional as F

class SoftmaxRegressionScratch:
    def __init__(self, input_dim, num_classes=10, learning_rate=0.01, max_epochs=30):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.W = torch.zeros((input_dim, num_classes), dtype=torch.float32)
        self.b = torch.zeros(num_classes, dtype=torch.float32)

    def softmax(self, z):
        exp_z = torch.exp(z - torch.max(z, dim=1, keepdim=True).values)
        return exp_z / torch.sum(exp_z, dim=1, keepdim=True)

    def cross_entropy_loss(self, y_pred, y_true):
        eps = 1e-8
        y_true_onehot = F.one_hot(y_true, num_classes=self.num_classes).float()
        return -torch.mean(torch.sum(y_true_onehot * torch.log(y_pred + eps), dim=1))

    def forward_pass(self, X_batch):
        scores = X_batch @ self.W + self.b
        return self.softmax(scores)

    def compute_accuracy(self, y_pred, y_true):
        preds = torch.argmax(y_pred, dim=1)
        return (preds == y_true).float().mean().item()

    def compute_gradients(self, X_batch, y_batch, y_pred):
        y_true_onehot = F.one_hot(y_batch, num_classes=self.num_classes).float()
        error = (y_pred - y_true_onehot) / X_batch.shape[0]
        dW = X_batch.T @ error
        db = torch.sum(error, dim=0)
        return dW, db

    def update_weights(self, dW, db):
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def fit(self, train_loader, val_loader):
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(self.max_epochs):
            train_loss, train_acc = 0, 0
            for X_batch, y_batch in train_loader:
                y_pred = self.forward_pass(X_batch)
                loss = self.cross_entropy_loss(y_pred, y_batch)
                dW, db = self.compute_gradients(X_batch, y_batch, y_pred)
                self.update_weights(dW, db)
                train_loss += loss.item()
                train_acc += self.compute_accuracy(y_pred, y_batch)
            train_losses.append(train_loss / len(train_loader))
            train_accs.append(train_acc / len(train_loader))

            # Validation
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = self.forward_pass(X_batch)
                    loss = self.cross_entropy_loss(y_pred, y_batch)
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
            return torch.argmax(y_pred, dim=1)
