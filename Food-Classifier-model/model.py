import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split

# import wandb
import torch.nn as nn
import torchvision.models as models
from torch.optim import lr_scheduler
from torchvision.models import ResNet50_Weights

# uncomment wandb import and variables if you want to use it

# set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb.login()

# wandb.init(project="food-101")

# Hyperparameters

num_epochs = 45
num_classes = 101
batch_size = 128
learning_rate = 0.001

# setting hyperparameters for wandb
# config = wandb.config
# config.num_epochs = num_epochs
# config.num_classes = num_classes
# config.batch_size = batch_size
# config.learning_rate = learning_rate

# Data Augmentation
data_transforms = transforms.Compose(
    [
        transforms.CenterCrop(400),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.42714671, 0.42714637, 0.42714605], [0.2871592, 0.28715929, 0.28715924]
        ),
    ]
)
# Downloading the dataset
train_dataset = torchvision.datasets.Food101(
    root="./data", download=True, split="train", transform=transforms
)

test_dataset = torchvision.datasets.Food101(
    root="./data", download=False, split="test", transform=transforms
)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders using the subsets
train_loader = torch.utils.data.DataLoader(
    train_subset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,  # Usually, we don't shuffle the validation set
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)


# Custom Pretrained Model Architecture
class FoodClassifier(nn.Module):
    def __init__(self, num_classes=101):
        super(FoodClassifier, self).__init__()

        # Load the pre-trained ResNet-50 model
        resnet50 = models.resnet50(pretrained=True)

        # Fine-tuning: Unfreeze the last few layers
        for param in list(resnet50.parameters())[:-10]:
            param.requires_grad = False
        for param in list(resnet50.parameters())[-10:]:
            param.requires_grad = True

        num_ftrs = resnet50.fc.in_features

        # Custom classifier
        custom_classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, num_classes),
        )

        resnet50.fc = custom_classifier
        self.resnet50 = resnet50

    def forward(self, x):
        return self.resnet50(x)


# Create instance of CNN, and define optimizer and criterion

model = FoodClassifier(num_classes).to(device)

wandb.watch(model, log_freq=100)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training the model

# Initialize variables for early stopping
best_val_loss = float("inf")
early_stopping_counter = 0
early_stopping_limit = 5

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        # wandb.log({"Training Loss": loss.item()})
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step: [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(
        f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
    )
    # wandb.log({"Validation Loss": avg_val_loss, "Validation Accuracy": val_accuracy})

    # Learning rate scheduling
    scheduler.step()

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_limit:
            print("Early stopping reached. Stopping training.")
            break

    # Save checkpoint after each epoch
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, f"checkpoint_{epoch}.pth")
    # wandb.save(f'checkpoint_{epoch}.pth')

# Save the model
torch.save(model.state_dict(), "Umodelv2.pth")
# wandb.finish()
