import torch
import torchvision
from torchvision import transforms
import numpy as np

# download the dataset
train_dataset = torchvision.datasets.Food101(
    root="../data", download=True, split="train"
)
# load the dataset
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)

# Initialize variables for mean and standard deviation
mean = np.zeros(3)
std = np.zeros(3)
N = 0

# Loop over the dataset
for images, _ in train_loader:
    # Change the shape from [batch_size, 3, H, W] -> [batch_size*H*W, 3]
    reshaped_images = images.view(-1, 3)

    # Update total number of images
    N += reshaped_images.shape[0]

    # Update mean and std
    mean += reshaped_images.mean(dim=0).numpy()
    std += reshaped_images.std(dim=0).numpy()

# Finalize mean and std
mean /= len(train_loader)
std /= len(train_loader)

print("Mean:", mean)
print("Std:", std)
