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
print(pytreemanager.tree)
pytreemanager._alias('add')
print(pytreemanager.tree)
pytreemanager.prune(retained_depth=[2])
print(pytreemanager.tree)
pytreemanager._alias('remove')
print(pytreemanager.tree)
pytreemanager._alias('add')
print(pytreemanager.tree)
# pytreemanager._merge_nodes(pytreemanager.tree.root, 0, 2)
# pytreemanager._merge_nodes(pytreemanager.tree.root, 1, 2)
# pytreemanager._merge_nodes(pytreemanager.tree.root, 2, 3)
# pytreemanager._merge_nodes(pytreemanager.tree.root, 3, 4)
# pytreemanager._merge_nodes(pytreemanager.tree.root, 4, 5)
# pytreemanager._merge_nodes(pytreemanager.tree.root, 5, 6)
# pytreemanager._merge_nodes(pytreemanager.tree.root, 6, 6)
pytreemanager.merger_nodes({pytreemanager.tree.root:[(0, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 13)]}) # pytreemanager._expand_leaves() and pytreemanager._alias('remove') are included in the _merge_nodes() function
print(pytreemanager.tree)


# def func(tree_node): # a function that wraps the sigma_max_Jacobian
#     # get the arguments for the sigma_max_Jacobian
#     module = tree_node.info.head.module
#     _, batched_inputs_features = pytreemanager._get_attr(tree_node, 'batched_input_features')

#     batched_inputs_features = torch.unsqueeze(batched_inputs_features, 1) # notice that we have to expand the dimension of the input features even if it has an batch dimension,e.g., (256, 3, 32, 32) to (256, 1, 3, 32, 32), because of the vmap function in the sigma_max_Jacobian. 

#     # compute the maximal singular value of the Jacobian of the module w.r.t. batched_inputs
#     msv_dict = {}
#     msv_dict['msv'] = sigma_max_Jacobian(module, batched_inputs_features, DEVICE, iters=10, tol=1e-5)
#     return msv_dict

# # Training loop
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for i, (inputs, labels) in enumerate(train_loader):
#         inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
#         # Ensure inputs require gradients
#         inputs.requires_grad_(True)

#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if (i + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
#             running_loss = 0.0

#         pytreemanager.feature_tracker(inputs)
#         pytreemanager.update_info(func)

     
# # E valuate the model
# model.eval()
# correct = 0
# total = 0
# for inputs, labels in test_loader:
#     inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#     outputs = model(inputs)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()


# print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
