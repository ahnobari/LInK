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