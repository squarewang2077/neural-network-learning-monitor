import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.stats import pearsonr
import pandas as pd
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
model = DeepReLUNetwork(input_size=3*32*32, hidden_sizes=[256, 256], output_size=10)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Import the learning monitor
pytreemanager = PyTreeManager()
pytreemanager.build_tree(model)
print(pytreemanager.tree)
pytreemanager._alias('add')
pytreemanager.prune(retained_depth=[2])
print(pytreemanager.tree)
pytreemanager.merge_nodes({pytreemanager.tree.root: [(0,2), (3,4), (5, 6), (7, 7)]})
print(pytreemanager.tree)
pytreemanager.merge_nodes({pytreemanager.tree.root: [(0,1), (2,3)]})
print(pytreemanager.tree)


def get_msv(tree_node): # a function that wraps the sigma_max_Jacobian
    # get the arguments for the sigma_max_Jacobian
    module = tree_node.info.head.module
    _, batched_inputs_features = pytreemanager._get_attr(tree_node, 'batched_input_features')

    batched_inputs_features = torch.unsqueeze(batched_inputs_features, 1) # notice that we have to expand the dimension of the input features even if it has an batch dimension,e.g., (256, 3, 32, 32) to (256, 1, 3, 32, 32), because of the vmap function in the sigma_max_Jacobian. 

    # compute the maximal singular value of the Jacobian of the module w.r.t. batched_inputs
    msv_dict = {}
    msv_dict['msv'] = sigma_max_Jacobian(module, batched_inputs_features, DEVICE, iters=10, tol=1e-5)
    return msv_dict

def model_eval():
    # E valuate the model
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

class CrossCorr():
    
    def __init__(self):
        self.pre_weight = None # initialize the weight tracker
        self.pre_depth = -1 # initialize the depth tracker

        # initialize the result dictionary
        self.result = {
            'avg_a': [],
            'avg_b': [],
            'avg_rho': [],
            'avg_tau': [],

            'var_a': [],
            'var_b': [],
            'var_rho': [],
            'var_tau': []
        }

    def get_cross_corr(self, tree_node): # a function that wraps the cross_correlation
        module = tree_node.info.head.module
        current_depth = tree_node.info.head.depth

        if isinstance(module, nn.Linear):
            current_weight = module.weight
            if (self.pre_weight is not None) and (self.pre_depth == current_depth): # if the previous weight is not None and the current depth is the same as the previous depth, then we can compute the cross-correlation
                self._get_cross_corr(self.pre_weight, current_weight)

            else:            
                self._asign_none() # assign None to the result dictionary
        else:
            self._asign_none() # assign None to the result dictionary

        self.pre_weight = current_weight # update the weight tracker
        self.pre_depth = current_depth # update the depth tracker   

        return self.result

    def _asign_none(self):

        self.result['avg_a'] = None  
        self.result['avg_b'] = None
        self.result['avg_rho'] = None
        self.result['avg_tau'] = None                

        self.result['var_a'] = None
        self.result['var_b'] = None
        self.result['var_rho'] = None
        self.result['var_tau'] = None

    def _get_cross_corr(self, pre_weight, current_weight): # a function that compute cross_correlation

        sub_result = {
            'a': torch.tensor([]),
            'b': torch.tensor([]),
            'rho': torch.tensor([]),
            'tau': torch.tensor([])
        }

        for i in range(current_weight.shape[0]):
            for j in range(pre_weight.shape[1]):
                a, b, rho, tau = self.compute_rho_and_tau(current_weight[i], pre_weight[:, j])
                sub_result['a'] = torch.cat((sub_result['a'], a.unsqueeze(0)))
                sub_result['b'] = torch.cat((sub_result['b'], b.unsqueeze(0)))
                sub_result['rho'] = torch.cat((sub_result['rho'], rho.unsqueeze(0)))
                sub_result['tau'] = torch.cat((sub_result['tau'], tau.unsqueeze(0)))

        # Compute the mean of a, b, rho and tau 
        self.result['avg_a'] = sub_result['a'].mean().item()  
        self.result['avg_b'] = sub_result['b'].mean().item()
        self.result['avg_rho'] = sub_result['rho'].mean().item()
        self.result['avg_tau'] = sub_result['tau'].mean().item()                

        # Compute the variance of rho and tau
        self.result['var_a'] = sub_result['a'].var().item()
        self.result['var_b'] = sub_result['b'].var().item()
        self.result['var_rho'] = sub_result['rho'].var().item()
        self.result['var_tau'] = sub_result['tau'].var().item()


    @staticmethod            
    def compute_rho_and_tau(x, y):
        '''
        Compute the cosine similarity and Pearson correlation coefficient between x and y. 
        Args:
            x (torch.Tensor): input tensor x
            y (torch.Tensor): input tensor y
        Returns:
            a (float): the mean of x
            b (float): the mean of y
            rho (float): the cosine similarity
            tau (float): the Pearson correlation coefficient   
        '''
        assert len(x) == len(y), "x and y must have the same shape"
        num = len(x)
        rho = torch.sum(x * y) / (torch.norm(x) * torch.norm(y))
        a = x.sum() / torch.norm(x)
        b = y.sum() / torch.norm(y)
        tau = (num * rho - a * b) /((num - a**2).sqrt() * (num - b**2).sqrt()) 

        tau_2 = pearsonr(x, y)

        assert (tau.item() - tau_2[0].item()) < 10e-6, "tau is not equal to tau_2"

        return a, b, rho, tau


cross_corr = CrossCorr()

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

        pytreemanager.feature_tracker(inputs)
        pytreemanager.update_info(cross_corr.get_cross_corr)

     
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


corr_df = pd.DataFrame(self.corr_dict)
corr_df.to_csv('./log/results/pearsonr_resnet18_cifar10.csv', index=False)
