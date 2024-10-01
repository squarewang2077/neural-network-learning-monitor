import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from nnlm import *
import argparse 


def main():

    parser = argparse.ArgumentParser(description='train_attack_analysis_MLPs')

    parser.add_argument('--beta', type=float, default=0.0, # should be 1.0, 6.0, 12.0 
                        help='the beta of the TRADES loss')

    parser.add_argument('--corr', type=bool, default=False, 
                        help='the control of the correlate layer')

    args = parser.parse_args()

    print(args.beta)
    print(args.corr)

    # Hyperparameters
    batch_size = 256
    learning_rate = 0.0003
    num_epochs = 200
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Model setup ###
    # Define Neural Network
    model = CorrNet(input_size=3*32*32, hidden_size=256, output_size=10, correlate_layer=args.corr)
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
    criterion = TradesLoss(model=model, optimizer=optimizer, step_size=0.003, epsilon=1.0, perturb_steps=40, beta=args.beta)
    ### End of optimizer and Loss setup ###

    ### Evaluation setup ###
    # Define a class to compute the cross-correlation
    evaluator = Evaluator(model, train_loader, test_loader, DEVICE, 1000, attack_norm='L2', eps=1.0)

    # initliaze the cross-correlation class
    cross_corr = CrossCorr()
    ### End of evaluation setup ###

    ### Setup the storage ###
    # initialize the results for cross layer correlation
    layers_alias = pytreemanager.search_info(conditions={'depth': 1}, keys=[
            'alias'
        ])

    result_mean_tau = {}
    result_std_tau = {}
    for i, value in enumerate(layers_alias['alias']):
        result_mean_tau[f'{str(value)}-{i+1}'] = []
        result_std_tau[f'{str(value)}-{i+1}'] = []

    result_eval = {
        'train_acc': [],
        'test_acc': [],
        'robust_acc': []
    }

    # append the cross-correlation to the pytreemanager
    pytreemanager.append_info(cross_corr.get_cross_corr)
    cross_corr.reset_trackers()
    ### End of storage setup ###

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
        cross_corr.reset_trackers()
        epoch_result_clc = pytreemanager.search_info(conditions={'depth': 1}, keys=[
            'depth', 'alias', 'module',
            'mean_tau', 'std_tau'
        ])

        # update the mean and std of the cross-correlation
        for key, tau in zip(result_mean_tau.keys(), epoch_result_clc['mean_tau']):
            result_mean_tau[key].append(tau)        

        for key, tau in zip(result_std_tau.keys(), epoch_result_clc['std_tau']):
            result_std_tau[key].append(tau)        


        # evaluation of the neural network
        epoch_result_eval = evaluator.evaluate()
        result_eval['train_acc'].append(epoch_result_eval['train_acc'])
        result_eval['test_acc'].append(epoch_result_eval['test_acc'])
        result_eval['robust_acc'].append(epoch_result_eval['robust_acc'])
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Accuracy: {epoch_result_eval["train_acc"]:.4f}, Test Accuracy: {epoch_result_eval["test_acc"]:.4f}, Robust Accuracy: {epoch_result_eval["robust_acc"]:.4f}')


    # save the results
    result_eval_df = pd.DataFrame(result_eval)
    result_mean_tau_df = pd.DataFrame(result_mean_tau)
    result_std_tau_df = pd.DataFrame(result_std_tau)

    result_eval_df.to_csv(f'./log/eval/eval_MLP(256x2-relu6)_trades({args.beta})_init({args.corr}).csv', index=False)
    result_mean_tau_df.to_csv(f'./log/mean_tau/mean_tau_MLP(256x2-relu6)_trades({args.beta})_init({args.corr}).csv', index=False)
    result_std_tau_df.to_csv(f'./log/std_tau/std_tau_MLP(256x2-relu6)_trades({args.beta})_init({args.corr}).csv', index=False)


if __name__ == '__main__':
    main()