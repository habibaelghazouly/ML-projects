import torch
from ..NNs.helpers import save_checkpoint, load_checkpoint
def train_model_cnn(
    model,
    train_loader,
    val_loader,
    criterion=None,
    optimizer=None,
    epochs=10,
    device="cpu",
    early_stopping_patience=3,
    min_delta=0.001,
    checkpoint_path=None,
):
    model.to(device)
    
    # History dictionary
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct_train / total_train
        
        # ---- Validation ----
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = 100 * correct_val / total_val
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
        
        # ---- Early Stopping ----
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"New best model found (val_loss={best_val_loss:.4f})")
            if checkpoint_path:
                save_checkpoint(model, optimizer, epochs, history, checkpoint_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break
    
    return history