import torch
import numpy as np

def wolfe_line_search_batch(f, x, p, c1=1e-4, c2=0.9, max_iter=10, tau=0.5):
    '''
    Perform a line search to find a step size that satisfies the Wolfe conditions.
    :param f: Function that takes a batch of points x and returns a tuple (f(x), grad f(x))
    :param x: Batch of current points (tensor of shape [B, d]).
    :param p: Batch of search directions (tensor of shape [B, d]).
    :param c1: Armijo condition parameter.
    :param c2: Curvature condition parameter.
    :param max_iter: Maximum number of iterations.
    :return: Tuple of tensors (alpha, converged), where alpha is the step size that satisfies the Wolfe conditions and
             converged is a boolean tensor indicating which instances have converged.
    '''
    B, d = x.shape
    alpha = torch.ones([B]).to(x.device)
    converged = torch.zeros(B).to(torch.bool).to(x.device)

    # Compute function value and gradient at the current point x, outside the loop
    f_x, grad_f_x = f(x.reshape([B,-1,2]))
    f_x = f_x.reshape([B,1])
    grad_f_x = grad_f_x.reshape([B,d])

    for _ in range(max_iter):
        x_new = x + alpha[:, np.newaxis] * p
        f_x_new, grad_f_x_new = f(x_new.reshape([B,-1,2]))
        f_x_new = f_x_new.reshape([B,1])
        grad_f_x_new = grad_f_x_new.reshape([B,d])
        

        armijo_condition = f_x_new <= f_x + c1 * alpha[:, np.newaxis] * (grad_f_x * p).sum(axis=1, keepdims=True)
        curvature_condition = (grad_f_x_new * p).sum(axis=1, keepdims=True) >= c2 * (grad_f_x * p).sum(axis=1, keepdims=True)

        conditions_met = torch.logical_and(armijo_condition, curvature_condition).flatten()

        alpha[~conditions_met] *= tau
        converged = torch.logical_or(converged, conditions_met)

        if converged.all():
            break

    failed_to_converge = ~converged
    # if failed_to_converge.any():
    #     print("Warning: Line search failed to converge for some instances.")

    return alpha, converged

def bfgs_update_batch(H_k, s_k, y_k):
    """
    Perform a batched BFGS update of the inverse Hessian approximation.

    :param H_k: Batch of current inverse Hessian approximations (tensor of shape [B, d, d]).
    :param s_k: Batch of steps taken in parameter space (tensor of shape [B, d]).
    :param y_k: Batch of changes in the gradient (tensor of shape [B, d]).
    :return: Batch of updated inverse Hessian approximations.
    """
    B, d, _ = H_k.shape
    I = torch.eye(d, dtype=H_k.dtype, device=H_k.device).expand(B, d, d)
    
    rho_k = 1.0 / (y_k * s_k).sum(dim=1)
    term1 = s_k.unsqueeze(2) * y_k.unsqueeze(1)  # shape [B, d, d]
    term2 = y_k.unsqueeze(2) * s_k.unsqueeze(1)  # shape [B, d, d]

    H_kp1 = (I - term1 * rho_k.unsqueeze(-1).unsqueeze(-1)) @ H_k @ (I - term2 * rho_k.unsqueeze(-1).unsqueeze(-1)) + (s_k.unsqueeze(2) * s_k.unsqueeze(1)) * rho_k.unsqueeze(-1).unsqueeze(-1)
    return H_kp1

def Batch_BFGS(x0s, obj_fn, max_iter = 100, threshhold = 0.01, tol=1e-6, f_tol = 1e-5, tau=0.5, line_search_max_iter=10, progress= lambda x: x):
    '''
    Perform a batched BFGS optimization.
    :param x0s: Batch of initial points (tensor of shape [B, d]).
    :param obj_fn: Function that takes a batch of points x and returns a tuple (f(x), grad f(x)).
    :param max_iter: Maximum number of iterations.
    :param threshhold: Stop if the function value is below this threshold.
    :param tol: Stop if the norm of the gradient is below this threshold.
    :param f_tol: Stop if the function value changes by less than this threshold.
    :param progress: Function that takes a float progress value between 0 and 1.
    :return: Tuple of tensors (x, f), where x is the batch of final points and f is the batch of final function values.
    '''

    B = x0s.shape[0]
    d = x0s.shape[1] * 2


    H = torch.eye(d,device=x0s.device).repeat(B, 1, 1)
    
    x = x0s + 0

    f,nabla = obj_fn(x0s)
    f_old = f + 10

    nabla = nabla.reshape([B,-1])
    
    it = 0
    x_new = x + 0
    converged = torch.Tensor([False])
    
    while True:
        if it>=max_iter:
            break
        
        if torch.norm(nabla,dim=-1).max()<=tol:
            break
        
        it += 1
        progress([it/max_iter,f.min().item()])

        f,_ = obj_fn(x_new)

        # if torch.abs(f-f_old).max()<=f_tol:
        #     break

        f_old = f + 0
        
        if f.min()<=threshhold:
            break

        p = -torch.bmm(H,nabla.unsqueeze(-1)).squeeze() # search direction (Newton Method)
        
        alpha, converged = wolfe_line_search_batch(obj_fn, x.reshape([B,-1]), p, tau=tau, max_iter=line_search_max_iter)
        # alpha, converged = wolfe_line_search_batch_iter(obj_fn, obj_fn_b, x.reshape([B,-1]), p, tau=tau, max_iter=line_search_max_iter)

        x_new = x + (alpha.unsqueeze(-1) * p).reshape(x.shape)
        
        f_new,nabla_new = obj_fn(x_new)
        x_new[f_new>=1e6] = x[f_new>=1e6]
        f_new,nabla_new = obj_fn(x_new)
        
        nabla_new = nabla_new.reshape([B,-1])
        
        y = nabla_new - nabla

        del_x = (x_new - x).reshape([B,-1])

        H = bfgs_update_batch(H,del_x,y)
        
        nabla = nabla_new + 0
        x = x_new
    return x, f