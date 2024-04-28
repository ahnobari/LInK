import numpy as np
import torch

# Dyadic Solution Path Finder
def find_path(A, motor = [0,1], fixed_nodes=[0, 1]):
    '''
    This function finds the solution path of a dyadic mechanism.

    Parameters:
        A (np.array): Adjacency matrix of the mechanism.
        motor (list): motor nodes.
        fixed_nodes (list): List of fixed nodes.

    Returns:
        path (np.array): Solution path of the mechanism.
        status (bool): True if the mechanism is dyadic and has a solution path, False otherwise.
    '''
    
    path = []
    
    A,fixed_nodes,motor = np.array(A),np.array(fixed_nodes),np.array(motor)
    
    unkowns = np.array(list(range(A.shape[0])))
    knowns = np.concatenate([fixed_nodes,[motor[-1]]])
    
    unkowns = unkowns[np.logical_not(np.isin(unkowns,knowns))]

    
    counter = 0
    while unkowns.shape[0] != 0:

        if counter == unkowns.shape[0]:
            # Non dyadic or DOF larger than 1
            return [], False
        n = unkowns[counter]
        ne = np.where(A[n])[0]
        
        kne = knowns[np.isin(knowns,ne)]
#         print(kne.shape[0])
        
        if kne.shape[0] == 2:
            
            path.append([n,kne[0],kne[1]])
            counter = 0
            knowns = np.concatenate([knowns,[n]])
            unkowns = unkowns[unkowns!=n]
        elif kne.shape[0] > 2:
            #redundant or overconstraint
            return [], False
        else:
            counter += 1
    
    return np.array(path), True

# Dyadic Mechanism Sorting
def get_order(A, motor = [0,1], fixed_nodes=[0, 1]):
    '''
    This function sorts the mechanism based on the solution path.

    Parameters:
        A (np.array): Adjacency matrix of the mechanism.
        motor (list): motor nodes.
        fixed_nodes (list): List of fixed nodes.
    
    Returns:
        joint order (np.array): Sorted order of the joints in a mechanism.
    '''
    path, status = find_path(A, motor, fixed_nodes)
    fixed_nodes = np.array(fixed_nodes)
    if status:
        return np.concatenate([motor,fixed_nodes[fixed_nodes!=motor[0]],path[:,0]])
    else:
        raise Exception("Non Dyadic or Dof larger than 1")

def sort_mechanism(A, x0, motor = [0,1], fixed_nodes=[0, 1]):
    '''
    This function sorts the mechanism based on the solution path.

    Parameters:
        A (np.array): Adjacency matrix of the mechanism.
        x0 (np.array): Initial positions of the joints.
        motor (list): motor nodes.
        fixed_nodes (list): List of fixed nodes.
    
    Returns:
        A_s (np.array): Sorted adjacency matrix of the mechanism.
        x0 (np.array): Sorted initial positions of the joints.
        motor (np.array): Motor nodes.
        fixed_nodes (np.array): Fixed nodes.
        ord (np.array): Sorted order of the joints in a mechanism.
    '''
    ord = get_order(A, motor, fixed_nodes)

    n_t = np.zeros(A.shape[0])
    n_t[fixed_nodes] = 1

    A_s = A[ord,:][:,ord]
    n_t_s = n_t[ord]

    return A_s, x0[ord], np.array([0,1]), np.where(n_t_s)[0], ord

# Vectorized Dyadic Solver
def solve_rev_vectorized_batch_CPU(As,x0s,node_types,thetas):
    
    Gs = np.square((np.expand_dims(x0s,1) - np.expand_dims(x0s,2))).sum(-1)
    
    x = np.zeros([x0s.shape[0],x0s.shape[1],thetas.shape[0],2])
    
    x = x + np.expand_dims(node_types * x0s,2)
    
    m = x[:,0] + np.tile(np.expand_dims(np.swapaxes(np.concatenate([np.expand_dims(np.cos(thetas),0),np.expand_dims(np.sin(thetas),0)],0),0,1),0),[x0s.shape[0],1,1]) * np.expand_dims(np.expand_dims(np.sqrt(Gs[:,0,1]),-1),-1)

    m = np.expand_dims(m,1)
    m = np.pad(m,[[0,0],[1,x0s.shape[1]-2],[0,0],[0,0]],mode='constant')
    x += m
    
    for k in range(3,x0s.shape[1]):
        
        inds = np.argsort(As[:,k,0:k])[:,-2:]
        
        l_ijs = np.linalg.norm(x[np.arange(x0s.shape[0]),inds[:,0]] - x[np.arange(x0s.shape[0]),inds[:,1]], axis=-1)
        
        gik = np.sqrt(np.expand_dims(Gs[np.arange(x0s.shape[0]),inds[:,0],np.ones(shape=[x0s.shape[0]],dtype=int)*k],-1))
        gjk = np.sqrt(np.expand_dims(Gs[np.arange(x0s.shape[0]),inds[:,1],np.ones(shape=[x0s.shape[0]],dtype=int)*k],-1))
        
        cosphis = (np.square(l_ijs) + np.square(gik) - np.square(gjk))/(2 * l_ijs * gik)
        
        cosphis = np.where(np.tile(node_types[:,k],[1,thetas.shape[0]])==0.0,cosphis,np.zeros_like(cosphis))
                             
        x0i1 = x0s[np.arange(x0s.shape[0]),inds[:,0],np.ones(shape=[x0s.shape[0]]).astype(np.int32)]
        x0i0 = x0s[np.arange(x0s.shape[0]),inds[:,0],np.zeros(shape=[x0s.shape[0]]).astype(np.int32)]
        
        x0j1 = x0s[np.arange(x0s.shape[0]),inds[:,1],np.ones(shape=[x0s.shape[0]]).astype(np.int32)]
        x0j0 = x0s[np.arange(x0s.shape[0]),inds[:,1],np.zeros(shape=[x0s.shape[0]]).astype(np.int32)]
        
        x0k1 = x0s[:,k,1]
        x0k0 = x0s[:,k,0]
        
        s = np.expand_dims(np.sign((x0i1-x0k1)*(x0i0-x0j0) - (x0i1-x0j1)*(x0i0-x0k0)),-1)
        

        phi = s * np.arccos(cosphis)
        
        a = np.transpose(np.concatenate([np.expand_dims(np.cos(phi),0),np.expand_dims(-np.sin(phi),0)],0),axes=[1,2,0])
        b = np.transpose(np.concatenate([np.expand_dims(np.sin(phi),0),np.expand_dims(np.cos(phi),0)],0),axes=[1,2,0])

        R = np.einsum("ijk...->jki...", np.concatenate([np.expand_dims(a,0),np.expand_dims(b,0)],0))
        
        xi = x[np.arange(x0s.shape[0]),inds[:,0]]
        xj = x[np.arange(x0s.shape[0]),inds[:,1]]
        
        scaled_ij = (xj-xi)/np.expand_dims(l_ijs,-1) * np.expand_dims(gik,-1)
        
        x_k = np.squeeze(np.matmul(R, np.expand_dims(scaled_ij,-1))) + xi
        x_k = np.where(np.tile(np.expand_dims(node_types[:,k],-1),[1,thetas.shape[0],2])==0.0,x_k,np.zeros_like(x_k))

        x_k = np.expand_dims(x_k,1)
        x_k = np.pad(x_k,[[0,0],[k,x0s.shape[1]-k-1],[0,0],[0,0]],mode='constant')
        
        x += x_k
    return x

# Solve a single mechanism
def solve_mechanism(A, x0 , motor = [0,1], fixed_nodes=[0, 1], thetas = np.linspace(0,2*np.pi,200)):

    A,x0,motor,fixed_nodes,ord = sort_mechanism(A, x0, motor, fixed_nodes)
    n_t = np.zeros([A.shape[0],1])
    n_t[fixed_nodes] = 1

    A = np.expand_dims(A,0)
    x0 = np.expand_dims(x0,0)
    n_t = np.expand_dims(n_t,0)

    sol = solve_rev_vectorized_batch_CPU(A,x0,n_t,thetas)

    return sol[0], ord

# GPU Solvers
def solve_rev_vectorized_batch(As,x0s,node_types,thetas,distance_to_locking = False):
    
    Gs = torch.cdist(x0s,x0s)
    
    x = torch.zeros([x0s.shape[0],x0s.shape[1],thetas.shape[0],2]).to(As.device)
    
    x = x + torch.unsqueeze(node_types * x0s,2)
    
    m = x[:,0] + torch.tile(torch.unsqueeze(torch.transpose(torch.cat([torch.unsqueeze(torch.cos(thetas),0),torch.unsqueeze(torch.sin(thetas),0)],0),0,1),0),[x0s.shape[0],1,1]) * torch.unsqueeze(torch.unsqueeze(Gs[:,0,1],-1),-1)
    
    x[:,1,:,:] = m
    
    cos_list = []
    
    for k in range(3,x0s.shape[1]):
        
        inds = torch.argsort(As[:,k,0:k])[:,-2:]
        
        l_ijs = torch.linalg.norm(x[np.arange(x0s.shape[0]),inds[:,0]] - x[np.arange(x0s.shape[0]),inds[:,1]], dim=-1)
        
        gik = torch.unsqueeze(Gs[np.arange(x0s.shape[0]),inds[:,0],np.ones(shape=[x0s.shape[0]])*k],-1)
        gjk = torch.unsqueeze(Gs[np.arange(x0s.shape[0]),inds[:,1],np.ones(shape=[x0s.shape[0]])*k],-1)
        
        cosphis = (torch.square(l_ijs) + torch.square(gik) - torch.square(gjk))/(2 * l_ijs * gik)
        
        cosphis = torch.where(torch.tile(node_types[:,k],[1,thetas.shape[0]])==0.0,cosphis,torch.zeros_like(cosphis))
        
        cos_list.append(cosphis.unsqueeze(1))
        
        x0i1 = x0s[np.arange(x0s.shape[0]),inds[:,0],np.ones(shape=[x0s.shape[0]]).astype(np.int32)]
        x0i0 = x0s[np.arange(x0s.shape[0]),inds[:,0],np.zeros(shape=[x0s.shape[0]]).astype(np.int32)]
        
        x0j1 = x0s[np.arange(x0s.shape[0]),inds[:,1],np.ones(shape=[x0s.shape[0]]).astype(np.int32)]
        x0j0 = x0s[np.arange(x0s.shape[0]),inds[:,1],np.zeros(shape=[x0s.shape[0]]).astype(np.int32)]
        
        x0k1 = x0s[:,k,1]
        x0k0 = x0s[:,k,0]
        
        s = torch.unsqueeze(torch.sign((x0i1-x0k1)*(x0i0-x0j0) - (x0i1-x0j1)*(x0i0-x0k0)),-1)
        

        phi = s * torch.arccos(cosphis)
        
        a = torch.permute(torch.cat([torch.unsqueeze(torch.cos(phi),0),torch.unsqueeze(-torch.sin(phi),0)],0),dims=[1,2,0])
        b = torch.permute(torch.cat([torch.unsqueeze(torch.sin(phi),0),torch.unsqueeze(torch.cos(phi),0)],0),dims=[1,2,0])

        R = torch.einsum("ijk...->jki...", torch.cat([torch.unsqueeze(a,0),torch.unsqueeze(b,0)],0))
        
        xi = x[np.arange(x0s.shape[0]),inds[:,0]]
        xj = x[np.arange(x0s.shape[0]),inds[:,1]]
        
        scaled_ij = (xj-xi)/torch.unsqueeze(l_ijs,-1) * torch.unsqueeze(gik,-1)
        
        x_k = torch.squeeze(torch.matmul(R, torch.unsqueeze(scaled_ij,-1))) + xi
        x_k = torch.where(torch.tile(torch.unsqueeze(node_types[:,k],-1),[1,thetas.shape[0],2])==0.0,x_k,torch.zeros_like(x_k))
        x[:,k,:,:] += x_k
    
    if distance_to_locking:
        return x, torch.cat(cos_list,dim=1)
    else:
        return x