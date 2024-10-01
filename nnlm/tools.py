import torch
import torch.nn as nn
from torch import vmap
from torch.func import jvp, vjp
from torch.func import vmap, jacrev  
from scipy.stats import pearsonr
from autoattack import AutoAttack
import torch.nn.functional as F


class CorrNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, correlate_layer = True):
        super().__init__()

        # Define the layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu6 = nn.ReLU6()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Custom weight initialization
        if correlate_layer:
            self.initialize_weights()
    
    def initialize_weights(self):
        
        std = (2/self.fc1.in_features)**0.5
        # Create the identical column vector for the first layer
        identical_vector = torch.randn(self.fc1.out_features) * std  # Random vector for initialization
        
        # Set each column of the first layer to be the same as the vector
        self.fc1.weight = nn.Parameter(identical_vector.unsqueeze(1).repeat(1, self.fc1.in_features))
            
        # Set each row of the second layer to be the same as the vector
        self.fc2.weight = nn.Parameter(identical_vector.unsqueeze(0).repeat(self.fc2.out_features, 1))
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        return x

class DeepMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        super().__init__()
        
        self.flatten = nn.Flatten()
        # Define the layers
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'relu6':
                layers.append(nn.ReLU6())
            elif activation == 'linear':
                layers.append(nn.Identity())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError('Invalid activation function')
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        
        # Create a Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.model(x)
        return x 


def sigma_max_Jacobian(func, batched_x, device, iters=10, tol=1e-5):
    '''
    The function is to compute the maximal singler value of hte Jacobian of the func w.r.t. batched_x.

    Args: 
        func: the function to compute the Jacobian 
        batched_x: the batched input x for the function 
    Return: 
        batched_msv.to_list: the list of the all msvs computed for each item in the batch 
    '''
    
    # Helper function to get aggregation dimensions (all dims except batch)
    def get_aggr_dim(x):
        return list(range(1, x.dim()))
    
    # Move inputs to the selected device
    batched_x = batched_x.to(device)

    # Initialize random batched vector u and normalize
    batched_u = torch.rand_like(batched_x, device=device)
    # aggr_dim = get_aggr_dim(batched_u)
    batched_u /= torch.linalg.vector_norm(batched_u, dim=get_aggr_dim(batched_u), keepdim=True) # noramlized batched_u for each batch 

    # Batched version of the function
    batched_func = vmap(func)
    
    previous_batched_msv = None  # To track changes in MSV across iterations

    for i in range(iters):
        # Compute Jacobian-vector product (forward-mode)
        _, batched_v = jvp(batched_func, (batched_x,), (batched_u,)) # v = J_{func}(x)*u
        
        # Compute vector-Jacobian product (backward-mode)
        _, vjp_fn = vjp(batched_func, batched_x) # this line construct the vjp function 
        batched_u = vjp_fn(batched_v)[0] # u = v^T*J_{func}(x)
        
        # Compute L2 norms of u and v
        u_L2_norms = torch.linalg.vector_norm(batched_u, dim=get_aggr_dim(batched_u))
        v_L2_norms = torch.linalg.vector_norm(batched_v, dim=get_aggr_dim(batched_v))
        
        # Compute the maximum singular values (MSVs)
        batched_msv = (u_L2_norms / v_L2_norms)
        
        # Handle potential NaNs in MSV computation
        batched_msv = torch.nan_to_num(batched_msv, nan=0.0)
        
        # Normalize u and v for the next iteration
        batched_u /= u_L2_norms.view(-1, *([1] * (batched_u.dim() - 1)))
        batched_v /= v_L2_norms.view(-1, *([1] * (batched_v.dim() - 1)))
        
        # Stopping condition: Check for convergence based on relative error
        if previous_batched_msv is not None:
            relative_error = torch.abs(batched_msv - previous_batched_msv) / (previous_batched_msv + 1e-7)
            if torch.max(relative_error) < tol:
                break
        
        # Detach and store MSVs for the next iteration
        previous_batched_msv = batched_msv.detach()
    
    print(f'max error: {relative_error.max()}; mean error: {relative_error.mean()}')


    # Convert MSV tensor to list and return
    return batched_msv.tolist()

class PGDAttack:
    def __init__(self, model, device, epsilon=1, alpha=0.003, num_iter=40, norm='L2'):
        """
        Initialize the PGD attack class.
        
        Args:
            model: The neural network model to attack.
            device: The device to run the attack on (e.g., 'cpu' or 'cuda').
            epsilon: The maximum perturbation.
            alpha: The step size.
            num_iter: The number of iterations.
            norm: The norm to use for the attack ('Linf' or 'L2').
        """
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.norm = norm

    def perturb(self, inputs, labels):
        """
        Perform PGD attack on the inputs.
        
        Args:
            inputs: The original inputs.
            labels: The true labels for the inputs.
        
        Returns:
            adv_inputs: The adversarial examples.
        """
        adv_inputs = inputs.clone().detach().requires_grad_(True).to(self.device)
        
        for _ in range(self.num_iter):
            outputs = self.model(adv_inputs)
            loss = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(loss, adv_inputs, retain_graph=False, create_graph=False)[0]
            
            if self.norm == 'Linf':
                adv_inputs = adv_inputs + self.alpha * grad.sign()
                eta = torch.clamp(adv_inputs - inputs, min=-self.epsilon, max=self.epsilon)
            elif self.norm == 'L2':
                grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
                normalized_grad = grad / (grad_norm + 1e-8)
                adv_inputs = adv_inputs + self.alpha * normalized_grad
                eta = adv_inputs - inputs
                eta_norm = torch.norm(eta.view(eta.size(0), -1), dim=1).view(-1, 1, 1, 1)
                factor = torch.min(torch.ones_like(eta_norm), self.epsilon / eta_norm)
                eta = eta * factor
            else:
                raise ValueError("Norm not supported")
            
            adv_inputs = torch.clamp(inputs + eta, min=0, max=1).detach_()
            adv_inputs.requires_grad_(True)
        
        return adv_inputs

class Evaluator:
    def __init__(self, model, train_loader, test_loader, device, num_samples, attack_norm='L2', eps=None):
        '''
        Initialize the Evaluator class.
        
        Args:
            model: The neural network model to evaluate.
            train_loader: DataLoader for the training data.
            test_loader: DataLoader for the test data.
            device: The device to run the evaluation on (e.g., 'cpu' or 'cuda').
            num_samples: The number of samples to evaluate.
            attack_norm: The norm to use for the attack ('Linf' or 'L2').
            eps: The epsilon value for the attack.
        '''
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_samples = num_samples
        self.attack_norm = attack_norm
        self.eps = eps if eps is not None else (1.0 if attack_norm == 'L2' else 8 / 255)
        self.eval_results = {'train_acc': None, 'test_acc': None, 'robust_acc': None}

    def evaluate(self):
        '''
        Evaluate the model on train, test, and robust data.
        The attack used is AutoAttack with the following attacks: apgd-ce, apgd-t, fab-t, square.
        '''
        self.model.eval()
        self.eval_results['train_acc'] = self._evaluate_loader(self.train_loader)
        self.eval_results['test_acc'] = self._evaluate_loader(self.test_loader)
        self.eval_results['robust_acc'] = self._evaluate_robust_accuracy()

        return self.eval_results

    def _evaluate_loader(self, loader):
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(loader):
            if total >= self.num_samples:
                break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return 100 * correct / total

    # Replace the _pgd_attack method with the PGDAttack class
    def _evaluate_robust_accuracy(self):
        robust_correct = 0
        total_robust = 0

        pgd_attack = PGDAttack(self.model, self.device, epsilon=self.eps, alpha=self.eps / 10, num_iter=40, norm=self.attack_norm)

        for i, (inputs, labels) in enumerate(self.test_loader):
            if total_robust >= self.num_samples:
                break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Generate adversarial examples using PGD attack
            adv_inputs = pgd_attack.perturb(inputs, labels)
            
            with torch.no_grad():
                outputs = self.model(adv_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_robust += labels.size(0)
                robust_correct += (predicted == labels).sum().item()

        return 100 * robust_correct / total_robust
        

class TradesLoss:
    def __init__(self, model, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_2'):
        """
        TRADES adversarial loss implementation as a class
        :param model: neural network model
        :param optimizer: optimizer for updating the model
        :param step_size: step size for generating adversarial examples
        :param epsilon: maximum perturbation
        :param perturb_steps: number of perturbation steps for generating adversarial examples
        :param beta: balancing parameter between natural and robust accuracy
        :param distance: type of perturbation ('l_inf' or 'l_2')
        """
        self.model = model
        self.optimizer = optimizer
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.distance = distance

    def __call__(self, x_natural, y):
        """
        Calculate the TRADES loss
        :param x_natural: natural examples
        :param y: natural labels
        :return: trades loss
        """
        # Set model to evaluation mode to compute the initial prediction
        self.model.eval()

        # Generate adversarial example
        batch_size = len(x_natural)
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = F.kl_div(F.log_softmax(self.model(x_adv), dim=1),
                                       F.softmax(self.model(x_natural), dim=1), reduction='batchmean')
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif self.distance == 'l_2':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = F.kl_div(F.log_softmax(self.model(x_adv), dim=1),
                                       F.softmax(self.model(x_natural), dim=1), reduction='batchmean')
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                grad_norm = torch.norm(grad.view(batch_size, -1), dim=1).view(-1, 1, 1, 1)
                normalized_grad = grad / (grad_norm + 1e-8)
                x_adv = x_adv.detach() + self.step_size * normalized_grad
                delta = x_adv - x_natural
                delta_norm = torch.norm(delta.view(batch_size, -1), dim=1).view(-1, 1, 1, 1)
                factor = torch.min(torch.ones_like(delta_norm), self.epsilon / delta_norm)
                x_adv = x_natural + delta * factor
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise ValueError("Distance not supported")

        # Switch model to train mode
        self.model.train()

        # Calculate the loss
        natural_output = self.model(x_natural)
        loss_natural = F.cross_entropy(natural_output, y)
        loss_robust = F.kl_div(F.log_softmax(self.model(x_adv), dim=1),
                               F.softmax(self.model(x_natural), dim=1), reduction='batchmean')
        loss = loss_natural + self.beta * loss_robust

        return loss
    

class CrossCorr():    
    def __init__(self):
        self.pre_weight = None # initialize the weight tracker
        self.pre_depth = -1 # initialize the depth tracker

        # initialize the result dictionary
        self.result = {'mean_tau': [], 'std_tau': []}

    def reset_trackers(self):
        self.pre_weight = None
        self.pre_depth = -1

    def get_cross_corr(self, tree_node): # a function that wraps the cross_correlation
        module = tree_node.info.head.module
        current_depth = tree_node.info.head.depth

        # the condition that whether the tree_node is the leaf node
        

        if isinstance(module, nn.Linear):
            current_weight = module.weight  
            if (self.pre_weight is not None) and (self.pre_depth == current_depth): # if the previous weight is not None and the current depth is the same as the previous depth, then we can compute the cross-correlation
                self._compute_cross_corr(self.pre_weight, current_weight)

            else:            
                self._asign_none() # assign None to the result dictionary

            self.pre_weight = current_weight # update the weight tracker

        else:
            self._asign_none() # assign None to the result dictionary

        self.pre_depth = current_depth # update the depth tracker   

        return self.result

    def _asign_none(self):

        self.result['mean_tau'] = None
        self.result['std_tau'] = None

    def _compute_cross_corr(self, W1, W2): # a function to compute the cross-correlation
        
        W2 = W2 - W2.mean(dim=1, keepdim=True) # substract the mean of each row for W2
        W2 = W2 / W2.std(dim=1, keepdim=True) # standardize the rows of W1 by its standerd deviation
        
        W1 = W1 - W1.mean(dim=0, keepdim=True) # substract the mean of each column for W2
        W1 = W1 / W1.std(dim=0, keepdim=True) # standardize the columns of W2 by its standerd deviation

        # assert whether W1 and W2 can be dot producted 
        assert W1.shape[0] == W2.shape[1], 'The two matrices cannot be dot producted'

        # the number of the rows of W1 is the number of the rows of the cross-correlation matrix
        num = W2.shape[1]

        # compute the cross-correlation matrix
        mean_tau = torch.matmul(W2, W1).mean() / num
        std_tau = (torch.matmul(W2, W1)/num).std()

        # update the result dictionary
        self.result['mean_tau'] = mean_tau.item()
        self.result['std_tau'] = std_tau.item()

