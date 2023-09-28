import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn


with open("labels.txt", "r") as f:  # Replace with the actual path to your labels file
    labels = [line.strip() for line in f.readlines()]


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


# Function to load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image


# Function to predict the label of the image
def predict_image(model, image_tensor):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)  # Get the index of the max value
        return labels[predicted.item()]


# Initialize the model architecture
model = FoodClassifier()

# Load the state dictionary
model_path = "checkpoint_8.pth"  # Replace with your model path
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])

# Move model to CPU (this is optional if you've already loaded to CPU)
model = model.to("cpu")

# Define the same transforms used during training and validation
transform = transforms.Compose(
    [
        transforms.CenterCrop(400),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.42714671, 0.42714637, 0.42714605], [0.2871592, 0.28715929, 0.28715924]
        ),
    ]
)

# Load and preprocess the image
image_path = (
    "pan-seared-top-sirloin-steak-h-500x500.jpg"  # Replace with your image path
)
image_tensor = preprocess_image(image_path, transform)

# Predict the label of the image
predicted_label = predict_image(model, image_tensor)

# Display the image and prediction
plt.imshow(Image.open(image_path))
plt.title(f"Predicted Label: {predicted_label}")
plt.axis("off")
plt.show()
