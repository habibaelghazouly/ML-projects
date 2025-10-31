# Imports
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


# Function to preprocess MNIST dataset
def preprocess_mnist(
    root=".data", batch_size=64, augment=False, flatten=False, random_seed=42
):

    # Define transformations
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    if augment:
        train_transform = transforms.Compose(
            [transforms.RandomRotation(10), transforms.ToTensor()]
        )

    if flatten:
        train_transform = transforms.Compose(
            [train_transform, transforms.Lambda(lambda x: x.view(-1))]
        )
        test_transform = transforms.Compose(
            [test_transform, transforms.Lambda(lambda x: x.view(-1))]
        )

    # Load MNIST dataset
    train_dataset = MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = MNIST(
        root="./data", train=True, download=True, transform=test_transform
    )

    # Create train and validation splits
    targets = train_dataset.targets.numpy()
    indices = list(range(len(targets)))

    train_indx, temp_indx = train_test_split(
        indices, test_size=0.4, stratify=targets, random_state=random_seed
    )
    val_indx, test_indx = train_test_split(
        temp_indx, test_size=0.5, stratify=targets[temp_indx], random_state=random_seed
    )

    train_subset = Subset(train_dataset, train_indx)
    valid_subset = Subset(test_dataset, val_indx)
    test_subset = Subset(test_dataset, test_indx)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = preprocess_mnist(augment=True)
    images, labels = next(iter(train_loader))
    print(f"Train batch shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
