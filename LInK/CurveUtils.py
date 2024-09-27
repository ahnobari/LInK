import numpy as np
import torch
import torch.nn as nn

def uniformize(curves: torch.tensor, n: int = 200) -> torch.tensor:
    with torch.no_grad():
        l = torch.cumsum(torch.nn.functional.pad(torch.norm(curves[:,1:,:] - curves[:,:-1,:],dim=-1),[1,0,0,0]),-1)
        l = l/l[:,-1].unsqueeze(-1)
        
        sampling = torch.linspace(0,1,n).to(l.device).unsqueeze(0).tile([l.shape[0],1])
        end_is = torch.searchsorted(l,sampling)[:,1:]
        end_ids = end_is.unsqueeze(-1).tile([1,1,2])
        
        l_end = torch.gather(l,1,end_is)
        l_start = torch.gather(l,1,end_is-1)
        ws = (l_end - sampling[:,1:])/(l_end-l_start)
    
    end_gather = torch.gather(curves,1,end_ids)
    start_gather = torch.gather(curves,1,end_ids-1)
    
    uniform_curves = torch.cat([curves[:,0:1,:],(end_gather - (end_gather-start_gather)*ws.unsqueeze(-1))],1)

    return uniform_curves

def path_processor(path, min_dist=0.01):
    out_path = []
    for i in range(path.shape[0]-1):
        l = np.linalg.norm(path[i]-path[i+1])
        if l > min_dist:
            n_points = np.math.ceil(l/min_dist)-1
            n_v = path[i+1] - path[i]
            n_v = n_v/l
            out_path.append(path[i])
            for j in range(n_points):
                out_path.append(path[i] + min_dist * n_v * (j+1))
        else:
            out_path.append(path[i])
    return np.array(out_path)

def get_scales(curves):
    return torch.sqrt(torch.square(curves-curves.mean(1)).sum(-1).sum(-1)/curves.shape[1])

def find_transforms(in_curves, target_curve, objective_fn, n_angles=100):

    translations = in_curves.mean(1)
    # center curves
    curves = in_curves - translations.unsqueeze(1)
    
    # apply uniform scaling
    s = torch.sqrt(torch.square(curves).sum(-1).sum(-1)/in_curves.shape[1])
    curves = curves/s.unsqueeze(-1).unsqueeze(-1)

    # find best rotation for each curve
    # Brute force search for best rotation
    test_angles = torch.linspace(0, 2*np.pi, n_angles).to(curves.device)
    R = torch.zeros(n_angles,2,2).to(curves.device)
    R[:,0,0] = torch.cos(test_angles)
    R[:,0,1] = -torch.sin(test_angles)
    R[:,1,0] = torch.sin(test_angles)
    R[:,1,1] = torch.cos(test_angles)
    R = R.unsqueeze(0).tile([curves.shape[0],1,1,1])
    R = R.reshape(-1,2,2)

    # rotate curves
    curves = curves.unsqueeze(1).tile([1,n_angles,1,1])
    curves = curves.reshape(-1,curves.shape[-2],curves.shape[-1])

    curves = torch.bmm(R, curves.transpose(1,2)).transpose(1,2)

    # find best rotation by measuring cdist to target curve
    target_curve = target_curve.unsqueeze(0).tile([curves.shape[0],1,1])
    # cdist = torch.cdist(curves, target_curve)
    
    # # chamfer distance
    # cdist = cdist.min(-1).values.mean(-1) + cdist.min(-2).values.mean(-1)
    cdist = objective_fn(curves, target_curve)
    cdist = cdist.reshape(-1,n_angles)
    best_rot_idx = cdist.argmin(-1)
    best_rot = test_angles[best_rot_idx]

    return translations, s, best_rot

def apply_transforms(curves, translations, scales, rotations):
    curves = curves*scales.unsqueeze(-1).unsqueeze(-1)
    R = torch.zeros(rotations.shape[0],2,2).to(curves.device)
    R[:,0,0] = torch.cos(rotations)
    R[:,0,1] = -torch.sin(rotations)
    R[:,1,0] = torch.sin(rotations)
    R[:,1,1] = torch.cos(rotations)
    curves = torch.bmm(R, curves.transpose(1,2)).transpose(1,2)
    curves = curves + translations.unsqueeze(1)
    return curves


def get_graph(con_mat: np.ndarray):
    num_nodes = (con_mat.sum(axis=-1)>0).sum()
    graph_list = []
    for (i,j) in itertools.combinations(range(num_nodes),2):
        if con_mat[i,j] == True:
            graph_list.append(set([i,j]))
    
    return graph_list
            
def get_triangles(con_mat: np.ndarray):
    triangles = []
    graph_list = get_graph(con_mat)
    for links_combination in itertools.combinations(graph_list,3):
        if len(set.union(*links_combination)) == 3:
            triangles.append(links_combination)
            
    return triangles,graph_list            

def check_commons(tuple_1: tuple,tuple_2: tuple):
    id_list_2 = list(range(len(tuple_2)))
    common = False
    for (i,set_1) in enumerate(tuple_1):
        for (j,set_2) in enumerate(tuple_2):
            if set_1 == set_2:
                common = True
                id_list_2.remove(j)
    if common:
        return True,tuple_1+tuple(tuple_2[i] for i in id_list_2)
    else:
        return False,()

def check_make_extension(edge_list:list, body_list:list):
    elem_count_check = [len(body)>3 for body in body_list]
    need_updates = any(elem_count_check)
    if need_updates:
        body_list_nups = [body for (ecc,body) in zip(elem_count_check,body_list) if ecc]
        for body in body_list_nups:
            nodes = {node for el_set in body for node in el_set}
            new_edges = [{i,j} for (i,j) in itertools.combinations(nodes,2) if {i,j} not in edge_list]
            if len(new_edges)==0:
                return None
            else:
                edge_list = edge_list+new_edges
                return edge_list
    else:
        return None
    
def get_bodies(con_mat: np.ndarray,new_con_mat: np.ndarray):
    triangles, edge_list = get_triangles(con_mat)
    unmod_edge_list = edge_list.copy()
    if triangles:
        body_list = [triangles[0]]
        for triang in triangles[1:]:
            cumm_common = False
            for (i,body) in enumerate(body_list):
                common,body_tuple = check_commons(triang,body)
                if common:
                    cumm_common = True
                    body_list.remove(body)
                    body_list.append(body_tuple)
            if cumm_common == False:
                body_list.append(triang)
        for body in body_list:
            for elem in body:
                edge_list.remove(elem)
        
        body_list += [(elem,) for elem in edge_list]
        
        rval = check_make_extension(unmod_edge_list,body_list)
        if rval is not None:
            up_con_mat = con_mat.copy()
            for edge in rval:
                ij = tuple(edge)
                up_con_mat[ij]=True
                up_con_mat[ij[::-1]]=True
            
            num_bodies,num_nodes,body_list = get_bodies(up_con_mat,new_con_mat)
            return num_bodies,num_nodes,body_list
    else:
        body_list = [(elem,) for elem in edge_list]
    
    ## getting new connectivity matrix, node and body numbers
    for (bn,body) in enumerate(body_list):
        for (i,j) in body:
            new_con_mat[i,j] = bn+1
            new_con_mat[j,i] = bn+1
    num_bodies = len(body_list)
    num_nodes = np.sum(np.any(con_mat,axis=0))
    return num_bodies,num_nodes,body_list