import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg11
import numpy as np
from scipy.stats import pearsonr
import pandas as pd


# Hyperparameters
batch_size = 256
learning_rate = 0.0003
num_epochs = 200
lambda_jacobian = 0  # Regularization strength
# lambda_jacobian = 1e-4  # Regularization strength
p = 2  # L_p norm parameter
sample_size = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define neural network
# Define the neural network class
class DeepReLUNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepReLUNetwork, self).__init__()
        
        self.flatten = nn.Flatten()
        # Define the layers
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        
        # Create a Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.model(x)
        return x 


# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define ResNet18 model
model = DeepReLUNetwork(input_size=3*32*32, hidden_sizes=[512, 256, 256, 256, 256, 512], output_size=10)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to compute the Jacobian of the model
def compute_jacobian(model, inputs):
    inputs.requires_grad_(True)
    outputs = model(inputs)
    jacobians = []
    for i in range(outputs.size(1)):
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[:, i] = 1
        jacobian = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        jacobians.append(jacobian)
    jacobians = torch.stack(jacobians, dim=1)
    return jacobians

class Learning_Monitor:
    def __init__(self, model):
        self.model = model


# Training loop
learning_monitor = Learning_Monitor(model)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # Ensure inputs require gradients
        inputs.requires_grad_(True)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    # Compute and log Jacobian for the epoch
    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(train_loader))
        inputs = inputs.to(DEVICE)
    inputs.requires_grad_(True)  # Ensure inputs require gradients for Jacobian computation
    jacobian = compute_jacobian(model, inputs)
    jacobian_norm = torch.norm(jacobian.view(jacobian.size(0), -1), p=p, dim=1).mean().item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Jacobian L_{p} Norm: {jacobian_norm:.4f}')

    learning_monitor.deompose()

    
# Evaluate the model
model.eval()
correct = 0
total = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

eval_by_epoch.save()

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
