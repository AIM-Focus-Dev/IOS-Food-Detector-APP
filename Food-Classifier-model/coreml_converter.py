import torch
import torch.nn as nn
import torchvision.models as models
import coremltools as ct

# Define the model architecture
num_classes = 101


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


# Create an instance of the model
torch_model = FoodClassifier()
torch_model.eval()

# Load the trained weights
weights_path = "checkpoint_8.pth"
checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
torch_model.load_state_dict(checkpoint["model_state_dict"])


# Create an example input
example_input = torch.rand(1, 3, 400, 400)

# Trace the model
traced_model = torch.jit.trace(torch_model, example_input)

# Convert TorchScript model to Core ML
model = ct.convert(traced_model, inputs=[ct.TensorType(shape=example_input.shape)])

# Save the converted model
model.save("FoodCL.mlmodel")
