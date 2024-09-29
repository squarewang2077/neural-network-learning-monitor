import torch
import torch.nn as nn
from torch import vmap
from torch.func import jvp, vjp
from torch.func import vmap, jacrev  
from scipy.stats import pearsonr
from autoattack import AutoAttack
import torch.nn.functional as F


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

    def _evaluate_robust_accuracy(self):
        adversary = AutoAttack(self.model, norm=self.attack_norm, eps=self.eps, version='standard', verbose=True)
        adversary.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']

        robust_correct = 0
        total_robust = 0

        for i, (inputs, labels) in enumerate(self.test_loader):
            if total_robust >= self.num_samples:
                break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                adv_inputs = adversary.run_standard_evaluation(inputs, labels, bs=inputs.size(0))
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

            self.pre_weight = current_weight # update the weight tracker

        else:
            self._asign_none() # assign None to the result dictionary

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

        # for i in range(current_weight.shape[0]):
        #     for j in range(pre_weight.shape[1]):
        #         a, b, rho, tau = self.compute_rho_and_tau(current_weight[i], pre_weight[:, j])
        #         sub_result['a'] = torch.cat((sub_result['a'].to(a.device), a.unsqueeze(0)))
        #         sub_result['b'] = torch.cat((sub_result['b'].to(b.device), b.unsqueeze(0)))
        #         sub_result['rho'] = torch.cat((sub_result['rho'].to(rho.device), rho.unsqueeze(0)))
        #         sub_result['tau'] = torch.cat((sub_result['tau'].to(tau.device), tau.unsqueeze(0)))

        a, b, rho, tau = self.compute_rho_and_tau(current_weight, pre_weight)

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
    def compute_rho_and_tau(matrix_x, matrix_y):
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
        # check that matrix x and y can be dot product
        assert matrix_x.shape[1] == matrix_y.shape[0], "matrix_x and matrix_y must be compatible for dot product"
        # reshpare the matrix_x and matrix_y as one-dimension tensor
        x = matrix_x.view(-1)
        y = matrix_y.view(-1)

        # computation of cosine-similarity 
        rho = (x @ y) / (torch.norm(x) * torch.norm(y))        
        rho = torch.sum(x * y) / (torch.norm(x) * torch.norm(y))
        a = x.sum() / torch.norm(x)
        b = y.sum() / torch.norm(y)
        tau = (num * rho - a * b) /((num - a**2).sqrt() * (num - b**2).sqrt()) 

        tau_2 = pearsonr(x.clone().detach().cpu().numpy(), y.clone().detach().cpu().numpy())

        assert (tau.item() - tau_2[0].item()) < 10e-6, "tau is not equal to tau_2"

        return a, b, rho, tau


    @staticmethod            
    def compute_avg_rowvar(matrix_x):
        '''
        Under the assumptions that the element for each row of the matrix drawn from the same distribution but the elements from different row may not, compute the average 
        variance of the rows.
        
        Args:
            matrix_x (torch.Tensor): input 2D tensor
        
        Returns:
            avg_var (float): the average variance of the rows
        '''
        # Check whether matrix_x is a 2-dimension tensor
        assert matrix_x.dim() == 2, "matrix_x must be a 2D tensor"

        # Check whether matrix_x is a nn.Tensor
        assert isinstance(matrix_x, torch.Tensor), "matrix_x must be a torch.Tensor"

        # Compute the variance for each row of the matrix_x
        row_variances = torch.var(matrix_x, dim=1, unbiased=True)

        # Compute the average of the variances
        avg_rowvar = torch.mean(row_variances).item()

        return avg_rowvar