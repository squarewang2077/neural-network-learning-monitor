import torch
from torch import vmap
from torch.func import jvp, vjp
from torch.func import vmap, jacrev  

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



