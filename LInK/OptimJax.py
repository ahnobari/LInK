import jax
import numpy as np
from .Solver import solve_rev_vectorized_batch_jax as solve_rev_vectorized_batch

import torch
import torch.nn as nn
from .BFGS_jax import Batch_BFGS

import matplotlib.pyplot as plt
import time

from tqdm.autonotebook import trange, tqdm
from .Visulization import draw_mechanism
import gradio as gr

@torch.compile
def cosine_search(target_emb, atlas_emb, max_batch_size = 1000000, ids = None):

    z = nn.functional.normalize(target_emb.unsqueeze(0).tile([max_batch_size,1]))

    if ids is None:
        ids = torch.arange(atlas_emb.shape[0]).to(target_emb.device).long()
    
    sim = []
    for i in range(int(np.ceil(ids.shape[0]/max_batch_size))):
        z1 = atlas_emb[ids[i*max_batch_size:(i+1)*max_batch_size]]
        sim.append(nn.functional.cosine_similarity(z1,z[0:z1.shape[0]]))
    sim = torch.cat(sim,0)

    return ids[(-sim).argsort()], sim

@jax.jit
def cosine_search_jax(target_emb, atlas_emb, max_batch_size = 1000000, ids = None):
    
        z = target_emb/jax.numpy.linalg.norm(target_emb)
    
        if ids is None:
            ids = jax.numpy.arange(atlas_emb.shape[0])
        
        sim = []
        for i in range(int(np.ceil(ids.shape[0]/max_batch_size))):
            z1 = atlas_emb[ids[i*max_batch_size:(i+1)*max_batch_size]]
            sim.append(jax.numpy.sum(z1*z[None],axis=-1)/jax.numpy.linalg.norm(z1,axis=-1)/jax.numpy.linalg.norm(z))
        sim = jax.numpy.concatenate(sim,0)
    
        return ids[jax.numpy.argsort(-sim)], sim

def uniformize(curves, n):

    l = jax.numpy.cumsum(jax.numpy.pad(jax.numpy.linalg.norm(curves[:,1:,:] - curves[:,:-1,:],axis=-1),((0,0),(1,0))),axis=-1)
    l = l/l[:,-1].reshape(-1,1)

    sampling = jax.numpy.linspace(-1e-6,1-1e-6,n)
    end_is = jax.vmap(lambda a: jax.numpy.searchsorted(a.reshape(-1),sampling)[1:])(l)

    end_ids = end_is

    l_end = l[jax.numpy.arange(end_is.shape[0]).reshape(-1,1),end_is]
    l_start = l[jax.numpy.arange(end_is.shape[0]).reshape(-1,1),end_is-1]
    ws = (l_end - sampling[1:].reshape(1,-1))/(l_end-l_start)

    end_gather = curves[jax.numpy.arange(end_ids.shape[0]).reshape(-1,1),end_ids]
    start_gather = curves[jax.numpy.arange(end_ids.shape[0]).reshape(-1,1),end_ids-1]

    uniform_curves = jax.numpy.concatenate([curves[:,0:1,:],(end_gather - (end_gather-start_gather)*ws[:,:,None])],1)

    return uniform_curves

@jax.jit
def _euclidean_distance(x, y) -> float:
    dist = jax.numpy.sqrt(jax.numpy.sum((x - y) ** 2))
    return dist


@jax.jit
def cdist(a, b):
    """Jax implementation of :func:`scipy.spatial.distance.cdist`.

    Uses euclidean distance.

    Parameters
    ----------
    x
        Array of shape (n_cells_a, n_features)
    y
        Array of shape (n_cells_b, n_features)

    Returns
    -------
    dist
        Array of shape (n_cells_a, n_cells_b)
    """
    return jax.vmap(lambda x,y: jax.vmap(lambda x1: jax.vmap(lambda y1: _euclidean_distance(x1, y1))(y))(x))(a,b)

@jax.jit
def batch_chamfer_distance(c1,c2):

    d = cdist(c1,c2)
    id1 = d.argmin(1)
    id2 = d.argmin(2)

    d1 = jax.numpy.linalg.norm(c2 - c1[jax.numpy.arange(id1.shape[0]).reshape(-1,1),id1],axis=-1).mean(1)
    d2 = jax.numpy.linalg.norm(c1 - c2[jax.numpy.arange(id2.shape[0]).reshape(-1,1),id2],axis=-1).mean(1)

    return d1 + d2

@jax.jit
def batch_ordered_distance(c1,c2):
    
    C = cdist(c2,c1)
    
    row_ind = jax.numpy.arange(c2.shape[1])

    row_inds = row_ind[row_ind[:,None]-jax.numpy.zeros_like(row_ind)].T
    col_inds = row_ind[row_ind[:,None]-row_ind].T 
    
    col_inds_ccw = jax.numpy.copy(col_inds[:,::-1])
    
    row_inds = row_inds
    col_inds = col_inds
    col_inds_ccw = col_inds_ccw

    argmin_cw = jax.numpy.argmin(C[:,row_inds, col_inds].sum(2),axis=1)
    argmin_ccw = jax.numpy.argmin(C[:,row_inds, col_inds_ccw].sum(2),axis=1)
    
    col_ind_cw = col_inds[argmin_cw, :]
    col_ind_ccw = col_inds_ccw[argmin_ccw, :]

    ds_cw = jax.numpy.square(jax.numpy.linalg.norm(c2 - c1[jax.numpy.arange(col_ind_cw.shape[0]).reshape(-1,1),col_ind_cw],axis=-1)).mean(1)
    ds_ccw = jax.numpy.square(jax.numpy.linalg.norm(c2 - c1[jax.numpy.arange(col_ind_ccw.shape[0]).reshape(-1,1),col_ind_ccw],axis=-1)).mean(1)
    ds = jax.numpy.minimum(ds_cw,ds_ccw)
    
    return ds

@jax.jit
def ordered_objective_batch(c1,c2):
    C = cdist(c2,c1)
    
    row_ind = jax.numpy.arange(c2.shape[1])

    row_inds = row_ind[row_ind[:,None]-jax.numpy.zeros_like(row_ind)].T
    col_inds = row_ind[row_ind[:,None]-row_ind].T 
    
    col_inds_ccw = jax.numpy.copy(col_inds[:,::-1])
    
    row_inds = row_inds
    col_inds = col_inds
    col_inds_ccw = col_inds_ccw

    argmin_cw = jax.numpy.argmin(C[:,row_inds, col_inds].sum(2),axis=1)
    argmin_ccw = jax.numpy.argmin(C[:,row_inds, col_inds_ccw].sum(2),axis=1)
    
    col_ind_cw = col_inds[argmin_cw, :]
    col_ind_ccw = col_inds_ccw[argmin_ccw, :]

    ds_cw = jax.numpy.square(jax.numpy.linalg.norm(c2 - c1[jax.numpy.arange(col_ind_cw.shape[0]).reshape(-1,1),col_ind_cw],axis=-1)).mean(1)
    ds_ccw = jax.numpy.square(jax.numpy.linalg.norm(c2 - c1[jax.numpy.arange(col_ind_ccw.shape[0]).reshape(-1,1),col_ind_ccw],axis=-1)).mean(1)
    ds = jax.numpy.minimum(ds_cw,ds_ccw)
    
    return ds * 2 * jax.numpy.pi

@jax.jit
def find_transforms(in_curves, target_curve, n_angles=100):

    objective_fn = batch_ordered_distance
    translations = in_curves.mean(1)
    # center curves
    curves = in_curves - translations[:,None]
    
    # apply uniform scaling
    s = jax.numpy.sqrt(jax.numpy.square(curves).sum(-1).sum(-1)/in_curves.shape[1])
    curves = curves/s[:,None,None]

    # find best rotation for each curve
    test_angles = jax.numpy.linspace(0, 2*jax.numpy.pi, n_angles)
    R = jax.numpy.zeros([n_angles,2,2])
    R = jax.numpy.hstack([jax.numpy.cos(test_angles)[:,None],-jax.numpy.sin(test_angles)[:,None],jax.numpy.sin(test_angles)[:,None],jax.numpy.cos(test_angles)[:,None]]).reshape(-1,2,2)

    R = R[None,:,:,:].repeat(curves.shape[0],0)
    R = R.reshape(-1,2,2)

    # rotate curves
    curves = curves[:,None,:,:].repeat(n_angles,1)
    curves = curves.reshape(-1,curves.shape[-2],curves.shape[-1])

    curves = jax.numpy.transpose(jax.numpy.matmul(R, jax.numpy.transpose(curves,(0,2,1))),(0,2,1))

    # find best rotation by measuring cdist to target curve
    target_curve = target_curve[None,:,:].repeat(curves.shape[0],0)
    # cdist = torch.cdist(curves, target_curve)
    
    # # chamfer distance
    cdist = objective_fn(curves, target_curve)
    cdist = cdist.reshape(-1,n_angles)
    best_rot_idx = cdist.argmin(-1)
    best_rot = test_angles[best_rot_idx]

    return translations, s, best_rot

@jax.jit
def apply_transforms(curves, translations, scales, rotations):
    curves = curves*scales[:,None,None]
    R = jax.numpy.zeros([rotations.shape[0],2,2])
    R = jax.numpy.hstack([jax.numpy.cos(rotations)[:,None],-jax.numpy.sin(rotations)[:,None],jax.numpy.sin(rotations)[:,None],jax.numpy.cos(rotations)[:,None]]).reshape(-1,2,2)
    curves = jax.numpy.transpose(jax.numpy.matmul(R, jax.numpy.transpose(curves,(0,2,1))),(0,2,1))
    curves = curves + translations[:,None]
    return curves

def preprocess_curves(curves, n=200):
    
    # equidistant sampling (Remove Timing)
    curves = uniformize(curves,n)

    # center curves
    curves = curves - curves.mean(1)[:,None]
    
    # apply uniform scaling
    s = jax.numpy.sqrt(jax.numpy.square(curves).sum(-1).sum(-1)/n)[:,None,None]
    curves = curves/s

    # find the furthest point on the curve
    max_idx = jax.numpy.square(curves).sum(-1).argmax(axis=1)

    # rotate curves so that the furthest point is horizontal
    theta = -jax.numpy.arctan2(curves[jax.numpy.arange(curves.shape[0]),max_idx,1],curves[jax.numpy.arange(curves.shape[0]),max_idx,0])
    
    # normalize the rotation
    R = jax.numpy.hstack([jax.numpy.cos(theta)[:,None],-jax.numpy.sin(theta)[:,None],jax.numpy.sin(theta)[:,None],jax.numpy.cos(theta)[:,None]])
    R = R.reshape(-1,2,2)

    # curves = torch.bmm(R,curves.transpose(1,2)).transpose(1,2)
    curves = jax.numpy.transpose(jax.numpy.matmul(R, jax.numpy.transpose(curves,(0,2,1))),(0,2,1))

    return curves

@jax.jit
def get_scales(curves):
    return jax.numpy.sqrt(jax.numpy.square(curves-curves.mean(1)).sum(-1).sum(-1)/curves.shape[1])

@jax.jit
def blind_objective_batch(curve,As,x0s,node_types, curve_size = 200, thetas=jax.numpy.linspace(0.0,2*jax.numpy.pi,2000), CD_weight=1.0, OD_weight=1.0, idxs = None):
    
    curve = preprocess_curves(curve[None], curve_size)[0]
    
    sol = solve_rev_vectorized_batch(As,x0s,node_types,thetas)
    
    if idxs is None:
        idxs = (As.sum(-1)>0).sum(-1)-1
    current_sol = sol[jax.numpy.arange(sol.shape[0]),idxs]
    
    #find nans at axis 0 level
    good_idx = jax.numpy.logical_not(jax.numpy.isnan(current_sol.sum(-1).sum(-1)))
    best_matches_masked = current_sol * good_idx[:,None,None]
    current_sol_r_masked = current_sol * ~good_idx[:,None,None]
    current_sol = uniformize(current_sol, current_sol.shape[1])
    current_sol = current_sol * good_idx[:,None,None] + current_sol_r_masked

    dummy = uniformize(curve[None],thetas.shape[0])[0]
    best_matches_r_masked = dummy[None].repeat(best_matches_masked.shape[0],0) * ~good_idx[:,None,None]
    best_matches = best_matches_masked + best_matches_r_masked
    best_matches = uniformize(best_matches,curve.shape[0])
    
    tr,sc,an = find_transforms(best_matches,curve)
    tiled_curves = curve[None,:,:].repeat(best_matches.shape[0],0)
    tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
    
    OD = batch_ordered_distance(current_sol[:,jax.numpy.linspace(0,current_sol.shape[1]-1,tiled_curves.shape[1]).astype(int),:]/sc[:,None,None],tiled_curves/sc[:,None,None])
    CD = batch_chamfer_distance(current_sol/sc[:,None,None],tiled_curves/sc[:,None,None])

    objective_function = CD_weight * CD + OD_weight * OD

    return objective_function



def make_batch_optim_obj(curve,As,x0s,node_types,timesteps=2000, CD_weight=1.0, OD_weight=1.0, start_theta=0.0, end_theta=2*jax.numpy.pi):

    thetas = jax.numpy.linspace(start_theta,end_theta,timesteps)
    sol = solve_rev_vectorized_batch(As,x0s,node_types,thetas)
    
    idxs = (As.sum(-1)>0).sum(-1)-1
    best_matches = sol[jax.numpy.arange(sol.shape[0]),idxs]
    good_idx = jax.numpy.logical_not(jax.numpy.isnan(best_matches.sum(-1).sum(-1)))
    best_matches_masked = best_matches * good_idx[:,None,None]
    best_matches_r_masked = best_matches[good_idx][0][None].repeat(best_matches.shape[0],0) * ~good_idx[:,None,None]
    best_matches = best_matches_masked + best_matches_r_masked
    best_matches = uniformize(best_matches,curve.shape[0])
    
    tr,sc,an = find_transforms(best_matches,curve, )
    tiled_curves = curve[None,:,:].repeat(best_matches.shape[0],0)
    tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
    
    # def objective(x0s_current):
    #     current_x0 = x0s_current
    #     sol = solve_rev_vectorized_batch(As,current_x0,node_types,thetas)
    #     current_sol = sol[jax.numpy.arange(sol.shape[0]),idxs]
        
    #     #find nans at axis 0 level
    #     good_idx = jax.numpy.logical_not(jax.numpy.isnan(current_sol.sum(-1).sum(-1)))

    #     current_sol_r_masked = current_sol * ~good_idx[:,None,None]
    #     current_sol = uniformize(current_sol, current_sol.shape[1])
    #     current_sol = current_sol * good_idx[:,None,None] + current_sol_r_masked

    #     OD = batch_ordered_distance(current_sol[:,jax.numpy.linspace(0,current_sol.shape[1]-1,tiled_curves.shape[1]).astype(int),:]/sc[:,None,None],tiled_curves/sc[:,None,None])
    #     CD = batch_chamfer_distance(current_sol/sc[:,None,None],tiled_curves/sc[:,None,None])
    #     objective_function = CD_weight* CD + OD_weight * OD

    #     objective_function = jax.numpy.where(jax.numpy.isnan(objective_function),1e6,objective_function)

    #     return objective_function

    # def get_sum(x0s_current):
    #     obj = objective(x0s_current)
    #     return obj.sum(), obj
        
    # def final(x0s_current):
    #     fn = jax.jit(jax.value_and_grad(get_sum,has_aux=True))

    #     val,grad = fn(x0s_current)

    #     val = jax.numpy.nan_to_num(val[1],nan=1e6)
    #     grad = jax.numpy.nan_to_num(grad,nan=0)

    #     return val,grad
    
    fn = lambda x0s_current: final(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight, OD_weight)

    return fn

def objective(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight, OD_weight):
    current_x0 = x0s_current
    sol = solve_rev_vectorized_batch(As,current_x0,node_types,thetas)
    current_sol = sol[jax.numpy.arange(sol.shape[0]),idxs]
    
    #find nans at axis 0 level
    good_idx = jax.numpy.logical_not(jax.numpy.isnan(current_sol.sum(-1).sum(-1)))

    current_sol_r_masked = current_sol * ~good_idx[:,None,None]
    current_sol = uniformize(current_sol, current_sol.shape[1])
    current_sol = current_sol * good_idx[:,None,None] + current_sol_r_masked

    OD = batch_ordered_distance(current_sol[:,jax.numpy.linspace(0,current_sol.shape[1]-1,tiled_curves.shape[1]).astype(int),:]/sc[:,None,None],tiled_curves/sc[:,None,None])
    CD = batch_chamfer_distance(current_sol/sc[:,None,None],tiled_curves/sc[:,None,None])
    objective_function = CD_weight* CD + OD_weight * OD

    objective_function = jax.numpy.where(jax.numpy.isnan(objective_function),1e6,objective_function)

    return objective_function

def get_sum(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight, OD_weight):
    obj = objective(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight, OD_weight)
    return obj.sum(), obj

fn = jax.jit(jax.value_and_grad(get_sum,has_aux=True))

def final(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight, OD_weight):
    
    val,grad = fn(x0s_current, As, node_types, curve, thetas, idxs, sc, tiled_curves, CD_weight, OD_weight)

    val = jax.numpy.nan_to_num(val[1],nan=1e6)
    grad = jax.numpy.nan_to_num(grad,nan=0)

    return val,grad

def smooth_hand_drawn_curves(curves, n=200, n_freq=5):

    # equidistant sampling (Remove Timing)
    curves = uniformize(curves,n)

    # center curves
    curves = curves - curves.mean(1)[:,None]
    
    # apply uniform scaling
    s = jax.numpy.sqrt(jax.numpy.square(curves).sum(-1).sum(-1)/n)[:,None,None]
    curves = curves/s

    # reduce with fft
    curves = jax.numpy.concatenate([jax.numpy.real(jax.numpy.fft.ifft(jax.numpy.fft.fft(curves[:,:,0],axis=1)[:,0:n_freq],n=n,axis=1))[:,:,None],
                                    jax.numpy.real(jax.numpy.fft.ifft(jax.numpy.fft.fft(curves[:,:,1],axis=1)[:,0:n_freq],n=n,axis=1))[:,:,None]],axis=2)

    return preprocess_curves(curves,n)

def progerss_uppdater(x, prog=None):
    if prog is not None:
        prog.update(1)
        prog.set_postfix_str(f'Current Best CD: {x[1]:.7f}')

def demo_progress_updater(x, prog=None, desc = ''):
    if prog is not None:
        prog(x[0], desc=desc + f'Current Best CD: {x[1]:.7f}')

class PathSynthesis:
    def __init__(self, trainer_instance, curves, As, x0s, node_types, precomputed_emb=None, optim_timesteps=2000, top_n = 300, init_optim_iters = 10, top_n_level2 = 30, CD_weight = 1.0, OD_weight = 0.25, BFGS_max_iter = 100, n_repos=0, BFGS_lineserach_max_iter=10, BFGS_line_search_mult = 0.5, butterfly_gen=200, butterfly_pop=200, curve_size = 200, smoothing = True, n_freq = 5, device = None, sizes = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if precomputed_emb is None:
            self.precomputed_emb = trainer_instance.compute_embeddings_base(curves, 1000)
        else:
            self.precomputed_emb = precomputed_emb

        self.curve_size = curve_size
        self.models = trainer_instance
        self.BFGS_max_iter = BFGS_max_iter
        self.BFGS_lineserach_max_iter = BFGS_lineserach_max_iter
        self.BFGS_line_search_mult = BFGS_line_search_mult
        self.butterfly_gen = butterfly_gen
        self.butterfly_pop = butterfly_pop
        self.smoothing = smoothing
        self.n_freq = n_freq
        self.curves = curves
        self.As = As
        self.x0s = x0s
        self.node_types = node_types
        self.top_n = top_n
        self.optim_timesteps = optim_timesteps
        self.init_optim_iters = init_optim_iters
        self.top_n_level2 = top_n_level2
        self.CD_weight = CD_weight
        self.OD_weight = OD_weight
        self.n_repos = n_repos
        
        if sizes is not None:
            self.sizes = sizes
        else:
            self.sizes = (As.sum(-1)>0).sum(-1)
    
    def synthesize(self, target_curve, verbose=True, visualize=True, partial=False, max_size=20, save_figs=None):
        
        start_time = time.time()
        
        # target_curve = torch.tensor(target_curve).float().to(self.device)
        
        og_scale = get_scales(target_curve[None])[0]
        
        if partial:
            size = target_curve.shape[0]
            #fit an ellipse that passes through the first and last point and is centered at the mean of the curve
            center = (target_curve[-1] + target_curve[0])/2
            start_point = target_curve[-1]
            end_point = target_curve[0]
            a = jax.numpy.linalg.norm(start_point-center)
            b = jax.numpy.linalg.norm(end_point-center)
            start_angle = jax.numpy.arctan2(start_point[1]-center[1],start_point[0]-center[0])
            end_angle = jax.numpy.arctan2(end_point[1]-center[1],end_point[0]-center[0])
            
            angles = jax.numpy.linspace(start_angle,end_angle,self.curve_size)
            ellipse = jax.numpy.stack([center[0] + a*jax.numpy.cos(angles),center[1] + b*jax.numpy.sin(angles)],1)
            
            angles = jax.numpy.linspace(start_angle+2*np.pi,end_angle,self.curve_size)
            ellipse_2 = jax.numpy.stack([center[0] + a*jax.numpy.cos(angles),center[1] + b*jax.numpy.sin(angles)],1)
            
            #ellipse 1 length
            l_1 = jax.numpy.linalg.norm(ellipse - target_curve.mean(0) , axis=-1).sum()
            #ellipse 2 length
            l_2 = jax.numpy.linalg.norm(ellipse_2 - target_curve.mean(0),axis=-1).sum()
            
            if l_1 > l_2:
                target_curve = jax.numpy.concatenate([target_curve,ellipse],0)
            else:
                target_curve = jax.numpy.concatenate([target_curve,ellipse_2],0)
        
        target_curve_copy = preprocess_curves(target_curve[None], self.curve_size)[0]
        target_curve_ = jax.numpy.copy(target_curve)
        
        if self.smoothing:
            target_curve = smooth_hand_drawn_curves(target_curve[None], n=self.curve_size, n_freq=self.n_freq)[0]
        else:
            target_curve = preprocess_curves(target_curve[None],self.curve_size)[0]
        
        if partial:
            # target_curve_copy_ = preprocess_curves(target_curve_[:size][None], self.curve_size)[0]
            tr,sc,an = find_transforms(uniformize(target_curve_[None],self.curve_size),target_curve, )
            transformed_curve = apply_transforms(target_curve[None],tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve-end_point,axis=-1))
            target_curve = preprocess_curves(target_curve[:matched_point_idx+1][None],self.curve_size)[0]
            
            target_uni = jax.numpy.copy(target_curve_copy)
            
            tr,sc,an = find_transforms(uniformize(target_curve_[None],self.curve_size),target_uni, )
            transformed_curve = apply_transforms(target_uni[None],tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve-end_point,axis=-1))
            target_curve_copy_ = uniformize(target_curve_copy[:matched_point_idx+1][None],self.curve_size)[0]
        
        if verbose:
            print('Curve preprocessing done')
            if visualize:
                fig, axs = plt.subplots(1,2,figsize=(10,5))
                if partial:
                    axs[0].plot(target_curve_[:size][:,0],target_curve_[:size][:,1],color="indigo")
                else:
                    axs[0].plot(target_curve_copy[:,0],target_curve_copy[:,1],color="indigo")
                axs[0].set_title('Original Curve')
                axs[0].axis('equal')
                axs[0].axis('off')
                
                axs[1].plot(target_curve[:,0],target_curve[:,1],color="indigo")
                axs[1].set_title('Preprocessed Curve')
                axs[1].axis('equal')
                axs[1].axis('off')

                if save_figs is not None:
                    fig.savefig(save_figs + '_preprocessing.png')
                else:
                    plt.show()
        
        input_tensor = target_curve[None]
        batch_padd = preprocess_curves(self.curves[np.random.choice(self.curves.shape[0],255)])
        input_tensor = torch.tensor(np.concatenate([input_tensor,batch_padd],0)).float().to(self.device)
        target_emb = self.models.compute_embeddings_input(input_tensor, 1000)[0]
        target_emb = torch.tensor(target_emb).float().to(self.device)
        
        ids = torch.tensor(np.where(self.sizes <= max_size)[0]).to(self.device).long()
        idxs, sim = cosine_search(target_emb, self.precomputed_emb, ids=ids, max_batch_size=100000)
        idxs = idxs.detach().cpu().numpy()
        
        if verbose:
            #max batch size is 250
            tr,sc,an = [],[],[]
            for i in range(int(np.ceil(self.top_n*5/250))):
                tr_,sc_,an_ = find_transforms(self.curves[idxs[i*250:(i+1)*250]],target_curve_copy, )
                tr.append(tr_)
                sc.append(sc_)
                an.append(an_)
            tr = jax.numpy.concatenate(tr,0)
            sc = jax.numpy.concatenate(sc,0)
            an = jax.numpy.concatenate(an,0)
            tiled_curves = target_curve_copy[None].repeat(self.top_n*5,0)
            tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
            CD = batch_ordered_distance(tiled_curves/sc[:,None,None],self.curves[idxs[:self.top_n*5]]/sc[:,None,None])
            
            #get best matches index
            tid = jax.numpy.argsort(CD)[:self.top_n]
            
            print(f'Best ordered distance found in top {self.top_n} is {CD.min()}')
            
            if visualize:
                grid_size = int(np.ceil(self.top_n**0.5))
                fig, axs = plt.subplots(grid_size,grid_size,figsize=(10,10))
                for i in range(grid_size):
                    for j in range(grid_size):
                        if i*grid_size+j < self.top_n:
                            axs[i,j].plot(self.curves[idxs[tid][i*grid_size+j]][:,0],self.curves[idxs[tid][i*grid_size+j]][:,1], color='indigo')
                            axs[i,j].plot(tiled_curves[tid][i*grid_size+j][:,0],tiled_curves[tid][i*grid_size+j][:,1],color="darkorange",alpha=0.7)
                        axs[i,j].axis('off')
                        axs[i,j].axis('equal')


                if save_figs is not None:
                    fig.savefig(save_figs + '_retrieved.png')
                else:
                    plt.show()
        
        As = self.As[idxs[tid]]
        x0s = self.x0s[idxs[tid]]
        node_types = self.node_types[idxs[tid]]
        
        # if partial:
        #     obj = make_batch_optim_obj_partial(target_curve_copy, target_curve_copy_, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=0.25)
        # else:
        obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=self.OD_weight, CD_weight=self.CD_weight)

        if verbose:
            print('Starting initial optimization')
            prog = trange(self.init_optim_iters)
        else:
            prog = None
        
        x,f = Batch_BFGS(x0s, obj, max_iter=self.init_optim_iters, line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult, progress=lambda x: progerss_uppdater(x,prog),threshhold=0.001)
        
        # top n level 2
        top_n_2 = f.argsort()[:self.top_n_level2]
        As = As[top_n_2]
        x0s = x[top_n_2]
        node_types = node_types[top_n_2]
        
        # if partial:
        #     obj = make_batch_optim_obj_partial(target_curve_copy, target_curve_copy_, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=0.25)
        # else:
        obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=self.OD_weight, CD_weight=self.CD_weight)
        
        if verbose:
            print('Starting second optimization stage')
            prog2 = trange(self.BFGS_max_iter)
        else:
            prog2 = None
        
        for i in range(self.n_repos):
            x,f = Batch_BFGS(x0s, obj, max_iter=self.BFGS_max_iter//(self.n_repos+1), line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult, progress=lambda x: progerss_uppdater(x,prog2))
            if verbose:
                print('Re-Positioning')
            x0s = x
        
        x,f = Batch_BFGS(x0s, obj, max_iter=self.BFGS_max_iter - self.n_repos* self.BFGS_max_iter//(self.n_repos+1), line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult, progress=lambda x: progerss_uppdater(x,prog2))
        
        best_idx = f.argmin()

        end_time = time.time()
        
        if verbose:
            print('Total time taken(s):',end_time-start_time)
        
        if verbose:
            print('Best chamfer distance found is',f.min())
        
        if partial:
            target_uni = uniformize(target_curve_copy[None],self.optim_timesteps)[0]
            
            tr,sc,an = find_transforms(uniformize(target_curve_[None],self.optim_timesteps),target_uni, )
            transformed_curve = apply_transforms(target_uni[None],tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve-end_point,axis=-1))
            
            sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],jax.numpy.linspace(0,jax.numpy.pi*2,self.optim_timesteps))
            tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
            best_matches = sol[np.arange(sol.shape[0]),tid]
            original_match = jax.numpy.copy(best_matches)
            best_matches = uniformize(best_matches,self.optim_timesteps)
            
            tr,sc,an = find_transforms(best_matches,target_uni, )
            tiled_curves = uniformize(target_uni[:matched_point_idx][None],self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
            transformed_curve = tiled_curves[0]
            
            best_matches = get_partial_matches(best_matches,tiled_curves[0],)
            
            CD = batch_chamfer_distance(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
            OD = ordered_objective_batch(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
            
            st_id, en_id = get_partial_index(original_match,tiled_curves[0],)
            
            st_theta = jax.numpy.linspace(0,2*np.pi,self.optim_timesteps)[st_id].squeeze()
            en_theta = jax.numpy.linspace(0,2*np.pi,self.optim_timesteps)[en_id].squeeze()
            
            st_theta[st_theta>en_theta] = st_theta[st_theta>en_theta] - 2*np.pi
            
        else:
            sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],jax.numpy.linspace(0,jax.numpy.pi*2,self.optim_timesteps))
            tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
            best_matches = sol[np.arange(sol.shape[0]),tid]
            best_matches = uniformize(best_matches,self.optim_timesteps)
            target_uni = uniformize(target_curve_copy,self.optim_timesteps)[0]
            
            tr,sc,an = find_transforms(best_matches,target_uni, )
            tiled_curves = uniformize(target_curve_copy[None],self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
            transformed_curve = tiled_curves[0]
            
            CD = batch_chamfer_distance(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
            OD = ordered_objective_batch(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
            
            st_theta = 0.
            en_theta = np.pi*2
        
        if visualize:
            ax = draw_mechanism(As[best_idx],x[best_idx],np.where(node_types[best_idx])[0],[0,1],highlight=tid[0],solve=True, thetas=np.linspace(st_theta,en_theta,self.optim_timesteps))
            # ax.plot(best_matches[0].detach().cpu().numpy()[:,0],best_matches[0].detach().cpu().numpy()[:,1],color="darkorange")
            ax.plot(transformed_curve[:,0], transformed_curve[:,1], color="indigo", alpha=0.7, linewidth=2)

            if save_figs is not None:
                fig.savefig(save_figs + '_final_candidate.png')
            else:
                plt.show()
        
        if verbose:
            print(f'Final Chamfer Distance: {CD*og_scale:.7f}, Ordered Distance: {OD*(og_scale**2):.7f}')
        
        A = As[best_idx]
        x = x[best_idx]
        node_types = node_types[best_idx]
        
        n_joints = (A.sum(-1)>0).sum()
        
        A = A[:n_joints,:][:,:n_joints]
        x = x[:n_joints]
        node_types = node_types[:n_joints]
        
        transformation = [tr,sc,an]
        start_theta = st_theta
        end_theta = en_theta
        performance = [CD*og_scale,OD*(og_scale**2),og_scale]
        
        return [A,x,node_types, start_theta, end_theta, transformation], performance, transformed_curve
        # return As[best_idx].cpu().numpy(), x[best_idx].cpu().numpy(), node_types[best_idx].cpu().numpy(), [tr,sc,an], transformed_curve, best_matches[0].detach().cpu().numpy(), [CD.item()*og_scale,OD.item()*og_scale**2]

    def demo_sythesize_step_1(self, target_curve, partial=False):
        torch.cuda.empty_cache()
        start_time = time.time()
        
        target_curve = preprocess_curves(target_curve[None],self.curve_size)[0]
        
        og_scale = get_scales(target_curve[None])[0]
        
        size = target_curve.shape[0]
        if partial:
            #fit an ellipse that passes through the first and last point and is centered at the mean of the curve
            center = (target_curve[-1] + target_curve[0])/2
            start_point = target_curve[-1]
            end_point = target_curve[0]
            a = jax.numpy.linalg.norm(start_point-center)
            b = jax.numpy.linalg.norm(end_point-center)
            start_angle = jax.numpy.arctan2(start_point[1]-center[1],start_point[0]-center[0])
            end_angle = jax.numpy.arctan2(end_point[1]-center[1],end_point[0]-center[0])
            
            angles = jax.numpy.linspace(start_angle,end_angle,self.curve_size)
            ellipse = jax.numpy.stack([center[0] + a*jax.numpy.cos(angles),center[1] + b*jax.numpy.sin(angles)],1)
            
            angles = jax.numpy.linspace(start_angle+2*np.pi,end_angle,self.curve_size)
            ellipse_2 = jax.numpy.stack([center[0] + a*jax.numpy.cos(angles),center[1] + b*jax.numpy.sin(angles)],1)
            
            #ellipse 1 length
            l_1 = jax.numpy.linalg.norm(ellipse - target_curve.mean(0) , axis=-1).sum()
            #ellipse 2 length
            l_2 = jax.numpy.linalg.norm(ellipse_2 - target_curve.mean(0),axis=-1).sum()
            
            if l_1 > l_2:
                target_curve = jax.numpy.concatenate([target_curve,ellipse],0)
            else:
                target_curve = jax.numpy.concatenate([target_curve,ellipse_2],0)
        
        target_curve_copy = preprocess_curves(target_curve[None], self.curve_size)[0]
        target_curve_ = jax.numpy.copy(target_curve)
        
        if self.smoothing:
            target_curve = smooth_hand_drawn_curves(target_curve[None], n=self.curve_size, n_freq=self.n_freq)[0]
        else:
            target_curve = preprocess_curves(target_curve[None],self.curve_size)[0]
        
        if partial:
            # target_curve_copy_ = preprocess_curves(target_curve_[:size][None], self.curve_size)[0]
            tr,sc,an = find_transforms(uniformize(target_curve_[None],self.curve_size),target_curve, )
            transformed_curve = apply_transforms(target_curve[None],tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve-end_point,axis=-1))
            target_curve = preprocess_curves(target_curve[:matched_point_idx+1][None],self.curve_size)[0]
            
            target_uni = jax.numpy.copy(target_curve_copy)
            
            tr,sc,an = find_transforms(uniformize(target_curve_[None],self.curve_size),target_uni, )
            transformed_curve = apply_transforms(target_uni[None],tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve-end_point,axis=-1))
            target_curve_copy_ = uniformize(target_curve_copy[:matched_point_idx+1][None],self.curve_size)[0]
        else:
            target_curve_copy_ = jax.numpy.copy(target_curve_copy)
        
        fig1 = plt.figure(figsize=(5,5))
        if partial:
            plt.plot(target_curve_[:size][:,0],target_curve_[:size][:,1],color="indigo")
        else:
            plt.plot(target_curve_copy[:,0],target_curve_copy[:,1],color="indigo")
        plt.axis('equal')
        plt.axis('off')
        plt.title('Original Curve')
        
        fig2 = plt.figure(figsize=(5,5))
        plt.plot(target_curve[:,0],target_curve[:,1],color="indigo")
        plt.axis('equal')
        plt.axis('off')
        plt.title('Preprocessed Curve')
        
        #save all variables which will be used in the next step
        payload = [target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial,size]
        
        return payload, fig1, fig2
    
    def demo_sythesize_step_2(self, payload, max_size=20):
        target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial, size = payload
        
        input_tensor = target_curve[None]
        batch_padd = preprocess_curves(self.curves[np.random.choice(self.curves.shape[0],255)])

        input_tensor = torch.tensor(np.concatenate([input_tensor,batch_padd],0)).float().to(self.device)
        with torch.cuda.amp.autocast():
            target_emb = self.models.compute_embeddings_input(input_tensor, 1000)[0]
        # target_emb = torch.tensor(target_emb).float().to(self.device)
        
        ids =np.where(self.sizes <= max_size)[0]
        idxs, sim = cosine_search_jax(target_emb, self.precomputed_emb, ids=ids)
        # idxs = idxs.detach().cpu().numpy()
        
        #max batch size is 250
        tr,sc,an = [],[],[]
        for i in range(int(np.ceil(self.top_n*5/250))):
            tr_,sc_,an_ = find_transforms(self.curves[idxs[i*250:(i+1)*250]],target_curve_copy, )
            tr.append(tr_)
            sc.append(sc_)
            an.append(an_)
        tr = jax.numpy.concatenate(tr,0)
        sc = jax.numpy.concatenate(sc,0)
        an = jax.numpy.concatenate(an,0)
        tiled_curves = target_curve_copy[None].repeat(self.top_n*5,0)
        tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
        CD = batch_ordered_distance(tiled_curves/sc[:,None,None],self.curves[idxs[:self.top_n*5]]/sc[:,None,None])
        
        #get best matches index
        tid = jax.numpy.argsort(CD)[:self.top_n]
        
        grid_size = int(np.ceil(self.top_n**0.5))
        fig, axs = plt.subplots(grid_size,grid_size,figsize=(10,10))
        for i in range(grid_size):
            for j in range(grid_size):
                if i*grid_size+j < self.top_n:
                    axs[i,j].plot(self.curves[idxs[tid][i*grid_size+j]][:,0],self.curves[idxs[tid][i*grid_size+j]][:,1], color='indigo')
                    axs[i,j].plot(tiled_curves[tid][i*grid_size+j][:,0],tiled_curves[tid][i*grid_size+j][:,1],color="darkorange",alpha=0.7)
                axs[i,j].axis('off')
                axs[i,j].axis('equal')
                
        payload = [idxs, tid, target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial, size]
        
        return payload, fig
    
    def demo_sythesize_step_3(self, payload, progress=None):
        idxs, tid, target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial, size = payload
        
        As = self.As[idxs[tid]]
        x0s = self.x0s[idxs[tid]]
        node_types = self.node_types[idxs[tid]]
        
        obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=self.OD_weight, CD_weight=self.CD_weight)

        prog = None
        
        x,f = Batch_BFGS(x0s, obj, max_iter=self.init_optim_iters, line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult, progress=lambda x: demo_progress_updater(x,progress,desc='Stage 1: '),threshhold=0.001)
        
        # top n level 2
        top_n_2 = f.argsort()[:self.top_n_level2]
        As = As[top_n_2]
        x0s = x[top_n_2]
        node_types = node_types[top_n_2]
        
        # if partial:
        #     obj = make_batch_optim_obj_partial(target_curve_copy, target_curve_copy_, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=0.25)
        # else:
        obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types,timesteps=self.optim_timesteps,OD_weight=self.OD_weight, CD_weight=self.CD_weight)
        prog2 = None
        
        for i in range(self.n_repos):
            x,f = Batch_BFGS(x0s, obj, max_iter=self.BFGS_max_iter//(self.n_repos+1), line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult, threshhold=0.04, progress=lambda x: demo_progress_updater([x[0]/(self.n_repos+1) + i/(self.n_repos+1),x[1]],progress,desc='Stage 2: '))
            x0s = x
        
        x,f = Batch_BFGS(x0s, obj, max_iter=self.BFGS_max_iter - self.n_repos* self.BFGS_max_iter//(self.n_repos+1), line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult, threshhold=0.04, progress=lambda x: demo_progress_updater([x[0]/(self.n_repos+1) + self.n_repos/(self.n_repos+1),x[1]],progress,desc='Stage 2: '))
        
        best_idx = f.argmin()

        end_time = time.time()
        
        if partial:
            target_uni = uniformize(target_curve_copy[None],self.optim_timesteps)[0]
            
            tr,sc,an = find_transforms(uniformize(target_curve_[None],self.optim_timesteps),target_uni, )
            transformed_curve = apply_transforms(target_uni[None],tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve-end_point,axis=-1))
            
            sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],jax.numpy.linspace(0,jax.numpy.pi*2,self.optim_timesteps))
            tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
            best_matches = sol[np.arange(sol.shape[0]),tid]
            original_match = jax.numpy.copy(best_matches)
            best_matches = uniformize(best_matches,self.optim_timesteps)
            
            tr,sc,an = find_transforms(best_matches,target_uni, )
            tiled_curves = uniformize(target_uni[:matched_point_idx][None],self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
            transformed_curve = tiled_curves[0]
            
            best_matches = get_partial_matches(best_matches,tiled_curves[0],)
            
            CD = batch_chamfer_distance(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
            OD = ordered_objective_batch(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
            
            st_id, en_id = get_partial_index(original_match,tiled_curves[0],)
            
            st_theta = np.linspace(0,2*np.pi,self.optim_timesteps)[st_id].squeeze()
            en_theta = np.linspace(0,2*np.pi,self.optim_timesteps)[en_id].squeeze()
            
            st_theta[st_theta>en_theta] = st_theta[st_theta>en_theta] - 2*np.pi
            
        else:
            sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],jax.numpy.linspace(0,jax.numpy.pi*2,self.optim_timesteps))
            tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
            best_matches = sol[np.arange(sol.shape[0]),tid]
            best_matches = uniformize(best_matches,self.optim_timesteps)
            target_uni = uniformize(target_curve_copy[None],self.optim_timesteps)[0]
            
            tr,sc,an = find_transforms(best_matches,target_uni, )
            tiled_curves = uniformize(target_curve_copy[None],self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
            transformed_curve = tiled_curves[0]
            
            CD = batch_chamfer_distance(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
            OD = ordered_objective_batch(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
            
            st_theta = 0.
            en_theta = np.pi*2
        
        n_joints = (As[best_idx].sum(-1)>0).sum()
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        ax = draw_mechanism(As[best_idx][:n_joints,:][:,:n_joints],x[best_idx][0:n_joints],np.where(node_types[best_idx][0:n_joints])[0],[0,1],highlight=tid[0].item(),solve=True, thetas=np.linspace(st_theta,en_theta,self.optim_timesteps),ax=ax)
        ax.plot(transformed_curve[:,0], transformed_curve[:,1], color="indigo", alpha=0.7, linewidth=2)
        
        A = As[best_idx]
        x = x[best_idx]
        node_types = node_types[best_idx]
        
        n_joints = (A.sum(-1)>0).sum()
        
        A = A[:n_joints,:][:,:n_joints]
        x = x[:n_joints]
        node_types = node_types[:n_joints]
        
        transformation = [tr,sc,an]
        start_theta = st_theta
        end_theta = en_theta
        performance = [CD.item()*og_scale,OD.item()*(og_scale**2),og_scale]
        torch.cuda.empty_cache()
        return fig, [[A,x,node_types, start_theta, end_theta, transformation], performance, transformed_curve], gr.update(value = {"Progress":1.0})

def get_partial_matches(curves, target_curve):
    objective_fn = batch_ordered_distance
    start_point = target_curve[0]
    end_point = target_curve[-1]
    
    start_match_idx = np.linalg.norm(start_point-curves,axis=-1).argmin(-1)
    end_match_idx = np.linalg.norm(end_point-curves,axis=-1).argmin(-1)
    
    test_target = jax.numpy.concatenate([target_curve[None],target_curve[None]],0)
    
    curves_out = []

    for i in range(curves.shape[0]):
        if start_match_idx[i] < end_match_idx[i]:
            partial_1 = uniformize(curves[i][start_match_idx[i]:end_match_idx[i]+1][None],target_curve.shape[0])[0]
            partial_2 = uniformize(jax.numpy.concatenate([curves[i][end_match_idx[i]:],curves[i][:start_match_idx[i]+1]],0)[None],target_curve.shape[0])[0]
            partials = jax.numpy.concatenate([partial_1[None],partial_2[None]],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            idx = f.argmin()
            curves_out.append(uniformize(partials[idx].squeeze()[None],target_curve.shape[0])[0])
        else:
            partial_1 = uniformize(curves[i][end_match_idx[i]:start_match_idx[i]+1][None],target_curve.shape[0])[0]
            partial_2 = uniformize(jax.numpy.concatenate([curves[i][start_match_idx[i]:],curves[i][:end_match_idx[i]+1]],0)[None],target_curve.shape[0])[0]
            partials = jax.numpy.concatenate([partial_1[None],partial_2[None]],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            idx = f.argmin()
            curves_out.append(uniformize(partials[idx].squeeze()[None],target_curve.shape[0])[0])
            
    return np.array(curves_out)

def get_partial_index(curves, target_curve):
    objective_fn = batch_ordered_distance
    start_point = target_curve[0]
    end_point = target_curve[-1]
    
    start_match_idx = np.linalg.norm(start_point-curves,axis=-1).argmin(-1)
    end_match_idx = np.linalg.norm(end_point-curves,axis=-1).argmin(-1)
    
    
    actual_start = np.copy(start_match_idx)
    actual_end = np.copy(end_match_idx)
    
    test_target = jax.numpy.concatenate([target_curve[None],target_curve[None]],0)
    
    for i in range(curves.shape[0]):
        if start_match_idx[i] < end_match_idx[i]:
            partial_1 = uniformize(curves[i][start_match_idx[i]:end_match_idx[i]+1][None],target_curve.shape[0])[0]
            partial_2 = uniformize(jax.numpy.concatenate([curves[i][end_match_idx[i]:],curves[i][:start_match_idx[i]+1]],0)[None],target_curve.shape[0])[0]
            partials = jax.numpy.concatenate([partial_1[None],partial_2[None]],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            idx = f.argmin()
            if idx == 1:
                actual_start[i], actual_end[i] = end_match_idx[i], start_match_idx[i]
        else:
            partial_1 = uniformize(curves[i][end_match_idx[i]:start_match_idx[i]+1][None],target_curve.shape[0])[0]
            partial_2 = uniformize(jax.numpy.concatenate([curves[i][start_match_idx[i]:],curves[i][:end_match_idx[i]+1]],0)[None],target_curve.shape[0])[0]
            partials = jax.numpy.concatenate([partial_1[None],partial_2[None]],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            idx = f.argmin()
            
            if idx == 0:
                actual_start[i], actual_end[i] = end_match_idx[i], start_match_idx[i]
    
    return actual_start, actual_end

def get_partial_matches_oto(curves, target_curves):
    objective_fn = batch_ordered_distance
    for i in range(curves.shape[0]):
        start_point = target_curves[i][0]
        end_point = target_curves[i][-1]
        
        start_match_idx = np.linalg.norm(start_point-curves[i:i+1],axis=-1).argmin(-1).squeeze()
        end_match_idx = np.linalg.norm(end_point-curves[i:i+1],axis=-1).argmin(-1).squeeze()
        
        test_target = jax.numpy.concatenate([target_curves[i][None],target_curves[i][None]],0)
        
        if start_match_idx < end_match_idx:
            partial_1 = uniformize(curves[i][start_match_idx:end_match_idx+1][None],curves.shape[1])[0]
            partial_2 = uniformize(jax.numpy.concatenate([curves[i][end_match_idx:],curves[i][:start_match_idx+1]],0)[None],curves.shape[1])[0]
            partials = jax.numpy.concatenate([partial_1[None],partial_2[None]],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            # idx = f.argmin()
            if f[0] < f[1]:
                curves[i] = uniformize(curves[i][start_match_idx:end_match_idx+1][None],curves.shape[1])[0]
            else:
                curves[i] = uniformize(jax.numpy.concatenate([curves[i][end_match_idx:],curves[i][:start_match_idx+1]],0)[None],curves.shape[1])[0]
            # curves[i] = uniformize(partials[idx].squeeze()[None],curves.shape[1])[0]
        else:
            partial_1 = uniformize(curves[i][end_match_idx:start_match_idx+1][None],curves.shape[1])[0]
            partial_2 = uniformize(jax.numpy.concatenate([curves[i][start_match_idx:],curves[i][:end_match_idx+1]],0)[None],curves.shape[1])[0]
            partials = jax.numpy.concatenate([partial_1[None],partial_2[None]],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            # idx = f.argmin()
            if f[0] < f[1]:
                curves[i] = uniformize(curves[i][end_match_idx:start_match_idx+1][None],curves.shape[1])[0]
            else:
                curves[i] = uniformize(jax.numpy.concatenate([curves[i][start_match_idx:],curves[i][:end_match_idx+1]],0)[None],curves.shape[1])[0]
            # curves[i] = uniformize(partials[idx].squeeze()[None],curves.shape[1])[0]
            
    return curves


