import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from nnlm import *

# Hyperparameters
batch_size = 256
learning_rate = 0.0003
num_epochs = 100
beta = 0.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tot_results = {        
        'depth' :[], 'alias' :[], 'module' :[],
        'avg_a' :[], 'avg_b' :[], 'avg_rho' :[], 'avg_tau' :[], 
        'var_a' :[], 'var_b' :[], 'var_rho' :[], 'var_tau' :[],
        'train_acc' :[], 'test_acc' :[], 'robust_acc' :[]
        }

### Model setup ###
# Define Neural Network
model = DeepMLP(input_size=3*32*32, hidden_sizes=[256, 256], output_size=10, activation='relu')
model = model.to(DEVICE)

# Model modification
pytreemanager = PyTreeManager()
pytreemanager.build_tree(model)
pytreemanager._alias('add')
pytreemanager.prune(retained_depth=[2])
print(pytreemanager.tree)
### End of model setup ###

### Data setup ###
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
### End of data setup ###

### Optimizer and Loss setup ###
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = TradesLoss(model=model, optimizer=optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=beta)
### End of optimizer and Loss setup ###

### Evaluation setup ###
# Define a class to compute the cross-correlation
evaluator = Evaluator(model, train_loader, test_loader, DEVICE, 1000, attack_norm='L2', eps=1.0)

# initliaze the cross-correlation class
cross_corr = CrossCorr()
### End of evaluation setup ###


### Training ###
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # Ensure inputs require gradients
        inputs.requires_grad_(True)

        # Forward pass
        outputs = model(inputs) # note the outputs here do not need to be feeded into the criterion
        loss = criterion(inputs, labels) # the model are trained with TRADES loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pytreemanager.update_info(cross_corr.get_cross_corr)
    clc_results = pytreemanager.search_info(conditions={'depth': 1}, keys=[
        'depth', 'alias', 'module',
        'avg_a', 'avg_b', 'avg_rho', 'avg_tau', 
        'var_a', 'var_b', 'var_rho', 'var_tau'
    ])
    eval_result = evaluator.evaluate()

    # update the clc_results and eval_results in the tot_results
    for key in clc_results:
        tot_results[key].append(clc_results[key])

    tot_results['train_acc'].append(eval_result['train_acc'])
    tot_results['test_acc'].append(eval_result['test_acc'])
    tot_results['robust_acc'].append(eval_result['robust_acc'])


corr_df = pd.DataFrame(tot_results)
corr_df.to_csv('./log/results/cross_layer_corr_MLP(256x2).csv', index=False)

