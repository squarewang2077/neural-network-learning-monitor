import torch
import unittest
from torch.nn import Linear, ReLU, Tanh
from nnlm.tools import *
from torch.func import vmap, jacrev


class TestSigmaMaxJacobian(unittest.TestCase):

    def test_msv_nonnegtive(self):
        """
        Test the sigma_max_Jacobian function with a simple linear transformation.
        """
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define a simple linear function f(x) = Wx, where W is a 3x3 matrix
        input_dim = 3
        output_dim = 3
        linear_layer = Linear(input_dim, output_dim)
        linear_layer = linear_layer.to(DEVICE)

        # Define a batch of input data (e.g., batch size of 5)
        batch_size = 5
        batched_input = torch.randn(batch_size, input_dim, device=DEVICE)

        # Call the sigma_max_Jacobian function
        max_singular_values = sigma_max_Jacobian(linear_layer, batched_input, DEVICE, iters=100, tol=1e-6)

        # Check that the output is a list of length batch_size
        self.assertEqual(len(max_singular_values), batch_size)

        # Check that the maximum singular values are non-negative
        for msv in max_singular_values:
            self.assertGreaterEqual(msv, 0)

    def test_identity_function(self):
        """
        Test the sigma_max_Jacobian function with an identity function. 
        The max singular value of the identity matrix should be 1.
        """
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define an identity function f(x) = x
        def identity_func(x):
            return x
        
        # Define a batch of input data (e.g., batch size of 5, input dimension 3)
        batch_size = 5
        input_dim = 3
        batched_input = torch.randn(batch_size, input_dim, device=DEVICE)
        
        # Call the sigma_max_Jacobian function
        max_singular_values = sigma_max_Jacobian(identity_func, batched_input, DEVICE, iters=100, tol=1e-6)
        
        # Check that the output is a list of length batch_size
        self.assertEqual(len(max_singular_values), batch_size)

        # For an identity function, the maximum singular value should be 1
        for msv in max_singular_values:
            self.assertAlmostEqual(msv, 1.0, places=5)

    def test_convergence(self):
        """
        Test that the function converges for a basic linear layer.
        """
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = 4
        output_dim = 4
        linear_layer = Linear(input_dim, output_dim)
        linear_layer = linear_layer.to(DEVICE)
        
        # Batch input of size 5
        batch_size = 5
        batched_input = torch.randn(batch_size, input_dim, device=DEVICE)

        # Call the sigma_max_Jacobian function with a high number of iterations
        max_singular_values = sigma_max_Jacobian(linear_layer, batched_input, DEVICE, iters=200, tol=1e-6)
        
        # Ensure the function returns a result and doesn't diverge
        self.assertIsNotNone(max_singular_values)
        
        # Ensure the length of the result matches the batch size
        self.assertEqual(len(max_singular_values), batch_size)
        
    def test_linear_function(self):
        """
        Test whether the output of sigma_max_Jacobian matches the maximal singular
        value computed by SVD using vmap and jacrev.
        Notice that for linear function, the Jacobian is unrelated with inputs, hence the output should all be the same 
        """
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define a simple linear function f(x) = Wx, where W is a 3x3 matrix
        input_dim = 3
        output_dim = 3
        linear_layer = Linear(input_dim, output_dim)
        linear_layer = linear_layer.to(DEVICE)

        # Define a batch of input data (e.g., batch size of 5)
        batch_size = 5
        batched_input = torch.randn(batch_size, input_dim, device=DEVICE)

        # Call the sigma_max_Jacobian function
        max_singular_values = sigma_max_Jacobian(linear_layer, batched_input, DEVICE, iters=100, tol=1e-6)

        # Function to compute SVD singular values for the Jacobian
        def compute_svd_jacobian(func, x):
            jacobian = jacrev(func)(x)  # Compute Jacobian
            jacobian = jacobian.view(jacobian.size(0), -1)  # Reshape into 2D
            singular_values = torch.linalg.svdvals(jacobian)  # Compute singular values
            return torch.max(singular_values)  # Return the max singular value

        # Vectorize the SVD computation over the batch
        svd_func = vmap(lambda x: compute_svd_jacobian(linear_layer, x))
        svd_singular_values = svd_func(batched_input)

        # Ensure both sigma_max_Jacobian and SVD give the same results
        for msv, svd_msv in zip(max_singular_values, svd_singular_values):
            self.assertAlmostEqual(msv, svd_msv.item(), places=4)

    def test_nonlinear_function(self):
        """
        Test whether the output of sigma_max_Jacobian matches the maximal singular
        value computed by SVD using vmap and jacrev for a non-linear model (ReLU).
        """
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define a non-linear model with ReLU
        class NonLinearModel(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super(NonLinearModel, self).__init__()
                self.linear = Linear(input_dim, output_dim)
                self.relu = ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))
        
        input_dim = 3
        output_dim = 3
        model = NonLinearModel(input_dim, output_dim).to(DEVICE)

        # Define a batch of input data (e.g., batch size of 5)
        batch_size = 5
        batched_input = torch.randn(batch_size, input_dim, device=DEVICE)

        # Call the sigma_max_Jacobian function
        max_singular_values = sigma_max_Jacobian(model, batched_input, DEVICE, iters=100, tol=1e-6)

        # Function to compute SVD singular values for the Jacobian
        def compute_svd_jacobian(func, x):
            jacobian = jacrev(func)(x)  # Compute Jacobian
            jacobian = jacobian.view(jacobian.size(0), -1)  # Reshape into 2D
            singular_values = torch.linalg.svdvals(jacobian)  # Compute singular values
            return torch.max(singular_values)  # Return the max singular value

        # Vectorize the SVD computation over the batch
        svd_func = vmap(lambda x: compute_svd_jacobian(model, x))
        svd_singular_values = svd_func(batched_input)

        # Ensure both sigma_max_Jacobian and SVD give the same results
        for msv, svd_msv in zip(max_singular_values, svd_singular_values):
            self.assertAlmostEqual(msv, svd_msv.item(), places=4)        

if __name__ == '__main__':
    unittest.main()


