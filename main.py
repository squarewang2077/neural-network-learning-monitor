import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from nnlm import *



# Hyperparameters
batch_size = 256
learning_rate = 0.0003
num_epochs = 200

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Define Neural Network
model = DeepReLUNetwork(input_size=3*32*32, hidden_sizes=[512, 256, 256, 256, 256, 512], output_size=10)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Import the learning monitor
pytreemanager = PyTreeManager()
pytreemanager.build_tree(model)
pytreemanager.prune(retained_depth=[2])
# pytreemanager._merge_nodes(pytreemanager.tree.root, 0, 1)
pytreemanager._alias('add')
print(pytreemanager.tree)
pytreemanager._merge_nodes(pytreemanager.tree.root, 0, 2)
print(pytreemanager.tree)
pytreemanager._merge_nodes(pytreemanager.tree.root, 1, 2)
print(pytreemanager.tree)
pytreemanager._merge_nodes(pytreemanager.tree.root, 2, 3)
print(pytreemanager.tree)
pytreemanager._merge_nodes(pytreemanager.tree.root, 3, 4)
print(pytreemanager.tree)
pytreemanager._merge_nodes(pytreemanager.tree.root, 4, 5)
print(pytreemanager.tree)
pytreemanager._merge_nodes(pytreemanager.tree.root, 5, 6)
print(pytreemanager.tree)

# use the traverse defined in the PyTreeManager to traverse the tree and compute the maximal singular value of the Jacobian of the module in the node and store the value in the first empty DLL node (the node with only .next and .prev attributes). Before the computation of the maximal singular value for the Jacobian of the module in the node, we first need to store the features after each module in the mpde and also stored in the first empty DLL node.
batched_inputs = None

# Initilize the DLL node to store the computed output features and msv
info_dll_node = DoublyListNode()
info_dll_node.batched_input_features = []
info_dll_node.batched_output_features = []
info_dll_node.msv = []
depth_tracker = 0 # the tracker of the depth for travsering the tree

def calculate_output_features(tree_node):
    # Track the change of the depth 
    current_depth = tree_node.info.head.depth

    # If detect the change of the depth, recover the batched_input_features to the inputs 
    if current_depth > depth_tracker:
        batched_input_features = batched_inputs 
        depth_tracker = current_depth
    else:
        # Pass the output features to the next node
        batched_input_features = batched_output_features

        

    # For each depth of the tree, compute the output features and msv    
    batched_output_features = tree_node.info.head.module(batched_input_features)
    msv = sigma_max_Jacobian(tree_node.info.head.module, batched_input_features, DEVICE)
    
    # Store the output features and msv in the first empty DLL node
    info_dll_node.batched_input_features.append(batched_input_features)
    info_dll_node.bathced_output_features.append(batched_output_features)    
    info_dll_node.msv.append(msv)

    # Append the info_dll_node to the tree_node
    tree_node.info.append(info_dll_node)


pytreemanager.tree.traverse(calculate_output_features)



# Training loop
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

        pytreemanager.tree.traverse()
    
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


print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
