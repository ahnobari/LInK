import numpy as np
import torch
import torch.nn as nn
from .Solver import solve_rev_vectorized_batch
from .CurveUtils import uniformize, find_transforms, apply_transforms, get_scales
from .DataUtils import preprocess_curves
import matplotlib.pyplot as plt
from .BFGS import Batch_BFGS
from tqdm.autonotebook import trange, tqdm
from .Visulization import draw_mechanism
from scipy.interpolate import CubicSpline
import time
import gradio as gr

def cosine_search(target_emb, atlas_emb, max_batch_size = 1000000):

    z = nn.functional.normalize(target_emb.unsqueeze(0).tile([max_batch_size,1]))

    sim = []
    for i in range(int(np.ceil(atlas_emb.shape[0]/max_batch_size))):
        z1 = atlas_emb[i*max_batch_size:(i+1)*max_batch_size]
        sim.append(nn.functional.cosine_similarity(z1,z[0:z1.shape[0]]))
    sim = torch.cat(sim,0)

    return(-sim).argsort(), sim

def batch_chamfer_distance(c1,c2):
    
    with torch.no_grad():
        d = torch.cdist(c1,c2)

        id1 = d.argmin(1)
        id2 = d.argmin(2)
    
    d1 = torch.linalg.norm(c2 - c1.gather(1,id1.unsqueeze(-1).tile([1,1,2])),dim=-1).mean(1)
    d2 = torch.linalg.norm(c1 - c2.gather(1,id2.unsqueeze(-1).tile([1,1,2])),dim=-1).mean(1)

    return d1 + d2

def batch_ordered_distance(c1,c2):
    with torch.no_grad():
        C = torch.cdist(c2,c1)
    
    row_ind = np.arange(c2.shape[1])

    row_inds = row_ind[row_ind[:,None]-np.zeros_like(row_ind)].T
    col_inds = row_ind[row_ind[:,None]-row_ind].T 
    
    col_inds_ccw = np.copy(col_inds[:,::-1])
    
    row_inds = torch.tensor(row_inds).to(c1.device)
    col_inds = torch.tensor(col_inds).to(c1.device)
    col_inds_ccw = torch.tensor(col_inds_ccw).to(c1.device)

    argmin_cw = torch.argmin(C[:,row_inds, col_inds].sum(2),dim=1)
    argmin_ccw = torch.argmin(C[:,row_inds, col_inds_ccw].sum(2),dim=1)
    
    col_ind_cw = col_inds[argmin_cw, :]
    col_ind_ccw = col_inds_ccw[argmin_ccw, :]
    
    ds_cw = torch.square(torch.linalg.norm(c2 - torch.gather(c1,1,col_ind_cw.unsqueeze(-1).repeat([1,1,2])),dim=-1)).mean(1)
    ds_ccw = torch.square(torch.linalg.norm(c2 - torch.gather(c1,1,col_ind_ccw.unsqueeze(-1).repeat([1,1,2])),dim=-1)).mean(1)
    
    ds = torch.minimum(ds_cw,ds_ccw)
    
    return ds


def ordered_objective_batch(c1,c2):
    with torch.no_grad():
        C = torch.cdist(c2,c1)
    
    row_ind = np.arange(c2.shape[1])

    row_inds = row_ind[row_ind[:,None]-np.zeros_like(row_ind)].T
    col_inds = row_ind[row_ind[:,None]-row_ind].T 
    
    col_inds_ccw = np.copy(col_inds[:,::-1])
    
    row_inds = torch.tensor(row_inds).to(c1.device)
    col_inds = torch.tensor(col_inds).to(c1.device)
    col_inds_ccw = torch.tensor(col_inds_ccw).to(c1.device)

    argmin_cw = torch.argmin(C[:,row_inds, col_inds].sum(2),dim=1)
    argmin_ccw = torch.argmin(C[:,row_inds, col_inds_ccw].sum(2),dim=1)
    
    col_ind_cw = col_inds[argmin_cw, :]
    col_ind_ccw = col_inds_ccw[argmin_ccw, :]
    
    ds_cw = torch.square(torch.linalg.norm(c2 - torch.gather(c1,1,col_ind_cw.unsqueeze(-1).repeat([1,1,2])),dim=-1)).mean(1)
    ds_ccw = torch.square(torch.linalg.norm(c2 - torch.gather(c1,1,col_ind_ccw.unsqueeze(-1).repeat([1,1,2])),dim=-1)).mean(1)
    
    ds = torch.minimum(ds_cw,ds_ccw)
    
    return ds * 2 * torch.pi

def make_batch_optim_obj(curve,As,x0s,node_types,timesteps=2000, CD_weight=1.0, OD_weight=1.0, start_theta=0.0, end_theta=2*np.pi):

    thetas = torch.linspace(start_theta,end_theta,timesteps).to(x0s.device)
    sol = solve_rev_vectorized_batch(As,x0s,node_types,thetas)
    
    idxs = (As.sum(-1)>0).sum(-1)-1
    best_matches = sol[np.arange(sol.shape[0]),idxs]
    good_idx = ~torch.isnan(best_matches.sum(-1).sum(-1))
    best_matches[~good_idx] = best_matches[good_idx][0]
    best_matches = uniformize(best_matches,curve.shape[0])
    
    
    tr,sc,an = find_transforms(best_matches,curve, batch_ordered_distance)
    tiled_curves = curve.unsqueeze(0).tile([best_matches.shape[0],1,1])
    tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
    
    def objective(x0s_current):
        with torch.enable_grad():
            current_x0 = torch.nn.Parameter(x0s_current,requires_grad = True).to(x0s_current.device)
            
            sol = solve_rev_vectorized_batch(As,current_x0,node_types,thetas)
            current_sol = sol[np.arange(sol.shape[0]),idxs]
            
            #find nans at axis 0 level
            good_idx = ~torch.isnan(current_sol.sum(-1).sum(-1))
            
            if good_idx.sum() == 0:
                return torch.tensor([1e6]*x0s_current.shape[0]).to(x0s_current.device), torch.zeros_like(x0s_current)
            
            current_sol[good_idx] = uniformize(current_sol[good_idx],current_sol.shape[1])
            
            OD = batch_ordered_distance(current_sol[:,np.linspace(0,current_sol.shape[1]-1,tiled_curves.shape[1],dtype=np.int32),:]/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            # OD.sum().backward()
            CD = batch_chamfer_distance(current_sol/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            objective_function = CD_weight* CD + OD_weight * OD
            objective_function.sum().backward()

            grad = current_x0.grad.detach()
        
        grad = grad * torch.logical_not(torch.isnan(CD)).unsqueeze(-1).unsqueeze(-1)
        objective_function = torch.nan_to_num(objective_function,nan=1e6)
        
        grad = torch.nan_to_num(grad)
        
        return objective_function, grad

    return objective

def make_batch_optim_obj_partial(curve, partial,As,x0s,node_types,timesteps=2000, CD_weight=1.0, OD_weight=1.0):

    thetas = torch.linspace(0,torch.pi*2,timesteps).to(x0s.device)
    sol = solve_rev_vectorized_batch(As,x0s,node_types,thetas)
    
    idxs = (As.sum(-1)>0).sum(-1)-1
    best_matches = sol[np.arange(sol.shape[0]),idxs]
    good_idx = ~torch.isnan(best_matches.sum(-1).sum(-1))
    best_matches[~good_idx] = best_matches[good_idx][0]
    best_matches = uniformize(best_matches,curve.shape[0])
    
    
    tr,sc,an = find_transforms(best_matches,curve, batch_ordered_distance)
    tiled_curves = partial.unsqueeze(0).tile([best_matches.shape[0],1,1])
    tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
    
    tiled_curves_uni = uniformize(tiled_curves,timesteps)
    
    def objective(x0s_current):
        with torch.enable_grad():
            current_x0 = torch.nn.Parameter(x0s_current,requires_grad = True).to(x0s_current.device)
            
            sol = solve_rev_vectorized_batch(As,current_x0,node_types,thetas)
            current_sol = sol[np.arange(sol.shape[0]),idxs]
            
            #find nans at axis 0 level
            good_idx = ~torch.isnan(current_sol.sum(-1).sum(-1))
            
            if good_idx.sum() == 0:
                return torch.tensor([1e6]*x0s_current.shape[0]).to(x0s_current.device), torch.zeros_like(x0s_current)
            
            current_sol[good_idx] = uniformize(current_sol[good_idx],current_sol.shape[1])
            current_sol[good_idx] = get_partial_matches_oto(current_sol[good_idx],tiled_curves_uni[good_idx],batch_ordered_distance)
            
            OD = batch_ordered_distance(current_sol[:,np.linspace(0,current_sol.shape[1]-1,tiled_curves.shape[1],dtype=np.int32),:]/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            # OD.sum().backward()
            CD = batch_chamfer_distance(current_sol/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            objective_function = CD_weight* CD + OD_weight * OD + torch.abs(current_x0).mean()
            objective_function.sum().backward()

            grad = current_x0.grad.detach()
        
        grad = grad * torch.logical_not(torch.isnan(CD)).unsqueeze(-1).unsqueeze(-1)
        objective_function = torch.nan_to_num(objective_function,nan=1e6)
        
        grad = torch.nan_to_num(grad)
        
        return objective_function, grad

    return objective

def smooth_hand_drawn_curves(curves, n=200, n_freq=5):

    curves = torch.tensor(curves).float()
    # equidistant sampling (Remove Timing)
    curves = uniformize(curves,n)

    # center curves
    curves = curves - curves.mean(1).unsqueeze(1)
    
    # apply uniform scaling
    s = torch.sqrt(torch.square(curves).sum(-1).sum(-1)/n).unsqueeze(-1).unsqueeze(-1)
    curves = curves/s

    # reduce with fft
    curves = torch.cat([torch.real(torch.fft.ifft(torch.fft.fft(curves[:,:,0],dim=1)[:,0:n_freq],n=n,dim=1)).unsqueeze(2),
                        torch.real(torch.fft.ifft(torch.fft.fft(curves[:,:,1],dim=1)[:,0:n_freq],n=n,dim=1)).unsqueeze(2)],dim=2)

    return preprocess_curves(curves,n)

def progerss_uppdater(x, prog=None):
    if prog is not None:
        prog.update(1)
        prog.set_postfix_str(f'Current Best CD: {x[1]:.7f}')

def demo_progress_updater(x, prog=None, desc = ''):
    if prog is not None:
        prog(x[0], desc=desc + f'Current Best CD: {x[1]:.7f}')

class PathSynthesis:
    def __init__(self, trainer_instance, curves, As, x0s, node_types, precomputed_emb=None, optim_timesteps=2000, top_n = 300, init_optim_iters = 10, top_n_level2 = 30, CD_weight = 1.0, OD_weight = 0.25, BFGS_max_iter = 100, n_repos=1, BFGS_lineserach_max_iter=10, BFGS_line_search_mult = 0.5, butterfly_gen=200, butterfly_pop=200, curve_size = 200, smoothing = True, n_freq = 5, device = None):
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
    
    def synthesize(self, target_curve, verbose=True, visualize=True, partial=False):
        
        start_time = time.time()
        
        target_curve = torch.tensor(target_curve).float().to(self.device)
        
        og_scale = get_scales(target_curve.unsqueeze(0))[0]
        
        if partial:
            size = target_curve.shape[0]
            #fit an ellipse that passes through the first and last point and is centered at the mean of the curve
            center = (target_curve[-1] + target_curve[0])/2
            start_point = target_curve[-1]
            end_point = target_curve[0]
            a = torch.linalg.norm(start_point-center)
            b = torch.linalg.norm(end_point-center)
            start_angle = torch.atan2(start_point[1]-center[1],start_point[0]-center[0])
            end_angle = torch.atan2(end_point[1]-center[1],end_point[0]-center[0])
            
            angles = torch.linspace(start_angle,end_angle,self.curve_size).to(self.device)
            ellipse = torch.stack([center[0] + a*torch.cos(angles),center[1] + b*torch.sin(angles)],1)
            
            angles = torch.linspace(start_angle+2*np.pi,end_angle,self.curve_size).to(self.device)
            ellipse_2 = torch.stack([center[0] + a*torch.cos(angles),center[1] + b*torch.sin(angles)],1)
            
            #ellipse 1 length
            l_1 = torch.linalg.norm(ellipse - target_curve.mean(0) , dim=-1).sum()
            #ellipse 2 length
            l_2 = torch.linalg.norm(ellipse_2 - target_curve.mean(0),dim=-1).sum()
            
            if l_1 > l_2:
                target_curve = torch.cat([target_curve,ellipse],0)
            else:
                target_curve = torch.cat([target_curve,ellipse_2],0)
        
        target_curve_copy = preprocess_curves(target_curve.unsqueeze(0), self.curve_size)[0]
        target_curve_ = target_curve.clone()
        
        if self.smoothing:
            target_curve = smooth_hand_drawn_curves(target_curve.unsqueeze(0), n=self.curve_size, n_freq=self.n_freq)[0]
        else:
            target_curve = preprocess_curves(target_curve.unsqueeze(0),self.curve_size)[0]
        
        if partial:
            # target_curve_copy_ = preprocess_curves(target_curve_[:size].unsqueeze(0), self.curve_size)[0]
            tr,sc,an = find_transforms(uniformize(target_curve_.unsqueeze(0),self.curve_size),target_curve, batch_ordered_distance)
            transformed_curve = apply_transforms(target_curve.unsqueeze(0),tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = torch.argmin(torch.linalg.norm(transformed_curve-end_point,dim=-1))
            target_curve = preprocess_curves(target_curve[:matched_point_idx+1].unsqueeze(0),self.curve_size)[0]
            
            target_uni = target_curve_copy.clone()
            
            tr,sc,an = find_transforms(uniformize(target_curve_.unsqueeze(0),self.curve_size),target_uni, batch_ordered_distance)
            transformed_curve = apply_transforms(target_uni.unsqueeze(0),tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = torch.argmin(torch.linalg.norm(transformed_curve-end_point,dim=-1))
            target_curve_copy_ = uniformize(target_curve_copy[:matched_point_idx+1].unsqueeze(0),self.curve_size)[0]
        
        if verbose:
            print('Curve preprocessing done')
            if visualize:
                fig, axs = plt.subplots(1,2,figsize=(10,5))
                if partial:
                    axs[0].plot(target_curve_[:size].detach().cpu().numpy()[:,0],target_curve_[:size].detach().cpu().numpy()[:,1],color="indigo")
                else:
                    axs[0].plot(target_curve_copy.cpu().numpy()[:,0],target_curve_copy.cpu().numpy()[:,1],color="indigo")
                axs[0].set_title('Original Curve')
                axs[0].axis('equal')
                axs[0].axis('off')
                
                axs[1].plot(target_curve.cpu().numpy()[:,0],target_curve.cpu().numpy()[:,1],color="indigo")
                axs[1].set_title('Preprocessed Curve')
                axs[1].axis('equal')
                axs[1].axis('off')
                plt.show()
        
        input_tensor = target_curve.unsqueeze(0)
        batch_padd = preprocess_curves(torch.tensor(self.curves[np.random.choice(self.curves.shape[0],255)]).float().to(self.device))
        input_tensor = torch.cat([input_tensor,batch_padd],0)
        target_emb = self.models.compute_embeddings_input(input_tensor, 1000)[0]
        target_emb = torch.tensor(target_emb).float().to(self.device)
        
        idxs, sim = cosine_search(target_emb, self.precomputed_emb)
        idxs = idxs.detach().cpu().numpy()
        
        if verbose:
            #max batch size is 250
            tr,sc,an = [],[],[]
            for i in range(int(np.ceil(self.top_n*5/250))):
                tr_,sc_,an_ = find_transforms(torch.tensor(self.curves[idxs[i*250:(i+1)*250]]).float().to(self.device),target_curve_copy, batch_ordered_distance)
                tr.append(tr_)
                sc.append(sc_)
                an.append(an_)
            tr = torch.cat(tr,0)
            sc = torch.cat(sc,0)
            an = torch.cat(an,0)
            tiled_curves = target_curve_copy.unsqueeze(0).tile([self.top_n*5,1,1])
            tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
            CD = batch_ordered_distance(tiled_curves/sc.unsqueeze(-1).unsqueeze(-1),torch.tensor(self.curves[idxs[:self.top_n*5]]).float().to(self.device)/sc.unsqueeze(-1).unsqueeze(-1))
            
            #get best matches index
            tid = torch.argsort(CD)[:self.top_n].detach().cpu().numpy()
            
            print(f'Best ordered distance found in top {self.top_n} is {CD.min()}')
            
            if visualize:
                grid_size = int(np.ceil(self.top_n**0.5))
                fig, axs = plt.subplots(grid_size,grid_size,figsize=(10,10))
                for i in range(grid_size):
                    for j in range(grid_size):
                        if i*grid_size+j < self.top_n:
                            axs[i,j].plot(self.curves[idxs[tid][i*grid_size+j]][:,0],self.curves[idxs[tid][i*grid_size+j]][:,1], color='indigo')
                            axs[i,j].plot(tiled_curves[tid][i*grid_size+j].cpu().numpy()[:,0],tiled_curves[tid][i*grid_size+j].cpu().numpy()[:,1],color="darkorange",alpha=0.7)
                        axs[i,j].axis('off')
                        axs[i,j].axis('equal')
                        
                plt.show()
        
        As = torch.tensor(self.As[idxs[tid]]).float().to(self.device)
        x0s = torch.tensor(self.x0s[idxs[tid]]).float().to(self.device)
        node_types = torch.tensor(self.node_types[idxs[tid]]).float().to(self.device)
        
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
            target_uni = uniformize(target_curve_copy.unsqueeze(0),self.optim_timesteps)[0]
            
            tr,sc,an = find_transforms(uniformize(target_curve_.unsqueeze(0),self.optim_timesteps),target_uni, batch_ordered_distance)
            transformed_curve = apply_transforms(target_uni.unsqueeze(0),tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = torch.argmin(torch.linalg.norm(transformed_curve-end_point,dim=-1))
            
            sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],torch.linspace(0,torch.pi*2,self.optim_timesteps).to(self.device))
            tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
            best_matches = sol[np.arange(sol.shape[0]),tid]
            original_match = best_matches.clone()
            best_matches = uniformize(best_matches,self.optim_timesteps)
            
            tr,sc,an = find_transforms(best_matches,target_uni, batch_ordered_distance)
            tiled_curves = uniformize(target_uni[:matched_point_idx].unsqueeze(0),self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
            transformed_curve = tiled_curves[0].detach().cpu().numpy()
            
            best_matches = get_partial_matches(best_matches,tiled_curves[0],batch_ordered_distance)
            
            CD = batch_chamfer_distance(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            OD = ordered_objective_batch(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            
            st_id, en_id = get_partial_index(original_match,tiled_curves[0],batch_ordered_distance)
            
            st_theta = torch.linspace(0,2*np.pi,self.optim_timesteps).to(self.device)[st_id].squeeze().cpu().numpy()
            en_theta = torch.linspace(0,2*np.pi,self.optim_timesteps).to(self.device)[en_id].squeeze().cpu().numpy()
            
            st_theta[st_theta>en_theta] = st_theta[st_theta>en_theta] - 2*np.pi
            
        else:
            sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],torch.linspace(0,torch.pi*2,self.optim_timesteps).to(self.device))
            tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
            best_matches = sol[np.arange(sol.shape[0]),tid]
            best_matches = uniformize(best_matches,self.optim_timesteps)
            target_uni = uniformize(target_curve_copy.unsqueeze(0),self.optim_timesteps)[0]
            
            tr,sc,an = find_transforms(best_matches,target_uni, batch_ordered_distance)
            tiled_curves = uniformize(target_curve_copy.unsqueeze(0),self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
            transformed_curve = tiled_curves[0].detach().cpu().numpy()
            
            CD = batch_chamfer_distance(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            OD = ordered_objective_batch(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            
            st_theta = 0.
            en_theta = np.pi*2
        
        if visualize:
            ax = draw_mechanism(As[best_idx].cpu().numpy(),x[best_idx].cpu().numpy(),np.where(node_types[best_idx].cpu().numpy())[0],[0,1],highlight=tid[0].item(),solve=True, thetas=np.linspace(st_theta,en_theta,self.optim_timesteps))
            # ax.plot(best_matches[0].detach().cpu().numpy()[:,0],best_matches[0].detach().cpu().numpy()[:,1],color="darkorange")
            ax.plot(transformed_curve[:,0], transformed_curve[:,1], color="indigo", alpha=0.7, linewidth=2)
            plt.show()
        
        if verbose:
            print(f'Final Chamfer Distance: {CD.item()*og_scale:.7f}, Ordered Distance: {OD.item()*(og_scale**2):.7f}')
        
        A = As[best_idx].cpu().numpy()
        x = x[best_idx].cpu().numpy()
        node_types = node_types[best_idx].cpu().numpy()
        
        n_joints = (A.sum(-1)>0).sum()
        
        A = A[:n_joints,:][:,:n_joints]
        x = x[:n_joints]
        node_types = node_types[:n_joints]
        
        transformation = [tr.cpu().numpy(),sc.cpu().numpy(),an.cpu().numpy()]
        start_theta = st_theta
        end_theta = en_theta
        performance = [CD.item()*og_scale,OD.item()*(og_scale**2)]
        
        return [A,x,node_types, start_theta, end_theta, transformation], performance, transformed_curve
        # return As[best_idx].cpu().numpy(), x[best_idx].cpu().numpy(), node_types[best_idx].cpu().numpy(), [tr,sc,an], transformed_curve, best_matches[0].detach().cpu().numpy(), [CD.item()*og_scale,OD.item()*og_scale**2]

    def demo_sythesize_step_1(self, target_curve, partial=False):
        torch.cuda.empty_cache()
        start_time = time.time()
        
        target_curve = preprocess_curves(torch.tensor(target_curve).float().to(self.device).unsqueeze(0),self.curve_size)[0]
        
        og_scale = get_scales(target_curve.unsqueeze(0))[0]
        
        size = target_curve.shape[0]
        if partial:
            #fit an ellipse that passes through the first and last point and is centered at the mean of the curve
            center = (target_curve[-1] + target_curve[0])/2
            start_point = target_curve[-1]
            end_point = target_curve[0]
            a = torch.linalg.norm(start_point-center)
            b = torch.linalg.norm(end_point-center)
            start_angle = torch.atan2(start_point[1]-center[1],start_point[0]-center[0])
            end_angle = torch.atan2(end_point[1]-center[1],end_point[0]-center[0])
            
            angles = torch.linspace(start_angle,end_angle,self.curve_size).to(self.device)
            ellipse = torch.stack([center[0] + a*torch.cos(angles),center[1] + b*torch.sin(angles)],1)
            
            angles = torch.linspace(start_angle+2*np.pi,end_angle,self.curve_size).to(self.device)
            ellipse_2 = torch.stack([center[0] + a*torch.cos(angles),center[1] + b*torch.sin(angles)],1)
            
            #ellipse 1 length
            l_1 = torch.linalg.norm(ellipse - target_curve.mean(0) , dim=-1).sum()
            #ellipse 2 length
            l_2 = torch.linalg.norm(ellipse_2 - target_curve.mean(0),dim=-1).sum()
            
            if l_1 > l_2:
                target_curve = torch.cat([target_curve,ellipse],0)
            else:
                target_curve = torch.cat([target_curve,ellipse_2],0)
        
        target_curve_copy = preprocess_curves(target_curve.unsqueeze(0), self.curve_size)[0]
        target_curve_ = target_curve.clone()
        
        if self.smoothing:
            target_curve = smooth_hand_drawn_curves(target_curve.unsqueeze(0), n=self.curve_size, n_freq=self.n_freq)[0]
        else:
            target_curve = preprocess_curves(target_curve.unsqueeze(0),self.curve_size)[0]
        
        if partial:
            # target_curve_copy_ = preprocess_curves(target_curve_[:size].unsqueeze(0), self.curve_size)[0]
            tr,sc,an = find_transforms(uniformize(target_curve_.unsqueeze(0),self.curve_size),target_curve, batch_ordered_distance)
            transformed_curve = apply_transforms(target_curve.unsqueeze(0),tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = torch.argmin(torch.linalg.norm(transformed_curve-end_point,dim=-1))
            target_curve = preprocess_curves(target_curve[:matched_point_idx+1].unsqueeze(0),self.curve_size)[0]
            
            target_uni = target_curve_copy.clone()
            
            tr,sc,an = find_transforms(uniformize(target_curve_.unsqueeze(0),self.curve_size),target_uni, batch_ordered_distance)
            transformed_curve = apply_transforms(target_uni.unsqueeze(0),tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = torch.argmin(torch.linalg.norm(transformed_curve-end_point,dim=-1))
            target_curve_copy_ = uniformize(target_curve_copy[:matched_point_idx+1].unsqueeze(0),self.curve_size)[0]
        else:
            target_curve_copy_ = target_curve_copy.clone()
        
        fig1 = plt.figure(figsize=(5,5))
        if partial:
            plt.plot(target_curve_[:size].detach().cpu().numpy()[:,0],target_curve_[:size].detach().cpu().numpy()[:,1],color="indigo")
        else:
            plt.plot(target_curve_copy.cpu().numpy()[:,0],target_curve_copy.cpu().numpy()[:,1],color="indigo")
        plt.axis('equal')
        plt.axis('off')
        plt.title('Original Curve')
        
        fig2 = plt.figure(figsize=(5,5))
        plt.plot(target_curve.cpu().numpy()[:,0],target_curve.cpu().numpy()[:,1],color="indigo")
        plt.axis('equal')
        plt.axis('off')
        plt.title('Preprocessed Curve')
        
        #save all variables which will be used in the next step
        payload = [target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial,size]
        
        return payload, fig1, fig2
    
    def demo_sythesize_step_2(self, payload):
        target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial, size = payload
        
        input_tensor = target_curve.unsqueeze(0)
        batch_padd = preprocess_curves(torch.tensor(self.curves[np.random.choice(self.curves.shape[0],255)]).float().to(self.device))
        input_tensor = torch.cat([input_tensor,batch_padd],0)
        target_emb = self.models.compute_embeddings_input(input_tensor, 1000)[0]
        target_emb = torch.tensor(target_emb).float().to(self.device)
        
        idxs, sim = cosine_search(target_emb, self.precomputed_emb)
        idxs = idxs.detach().cpu().numpy()
        
        #max batch size is 250
        tr,sc,an = [],[],[]
        for i in range(int(np.ceil(self.top_n*5/250))):
            tr_,sc_,an_ = find_transforms(torch.tensor(self.curves[idxs[i*250:(i+1)*250]]).float().to(self.device),target_curve_copy, batch_ordered_distance)
            tr.append(tr_)
            sc.append(sc_)
            an.append(an_)
        tr = torch.cat(tr,0)
        sc = torch.cat(sc,0)
        an = torch.cat(an,0)
        tiled_curves = target_curve_copy.unsqueeze(0).tile([self.top_n*5,1,1])
        tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
        CD = batch_ordered_distance(tiled_curves/sc.unsqueeze(-1).unsqueeze(-1),torch.tensor(self.curves[idxs[:self.top_n*5]]).float().to(self.device)/sc.unsqueeze(-1).unsqueeze(-1))
        
        #get best matches index
        tid = torch.argsort(CD)[:self.top_n].detach().cpu().numpy()
        
        grid_size = int(np.ceil(self.top_n**0.5))
        fig, axs = plt.subplots(grid_size,grid_size,figsize=(10,10))
        for i in range(grid_size):
            for j in range(grid_size):
                if i*grid_size+j < self.top_n:
                    axs[i,j].plot(self.curves[idxs[tid][i*grid_size+j]][:,0],self.curves[idxs[tid][i*grid_size+j]][:,1], color='indigo')
                    axs[i,j].plot(tiled_curves[tid][i*grid_size+j].cpu().numpy()[:,0],tiled_curves[tid][i*grid_size+j].cpu().numpy()[:,1],color="darkorange",alpha=0.7)
                axs[i,j].axis('off')
                axs[i,j].axis('equal')
                
        payload = [idxs, tid, target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial, size]
        
        return payload, fig
    
    def demo_sythesize_step_3(self, payload, progress=None):
        idxs, tid, target_curve_copy, target_curve_copy_, target_curve_, target_curve, og_scale, partial, size = payload
        
        As = torch.tensor(self.As[idxs[tid]]).float().to(self.device)
        x0s = torch.tensor(self.x0s[idxs[tid]]).float().to(self.device)
        node_types = torch.tensor(self.node_types[idxs[tid]]).float().to(self.device)
        
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
            x,f = Batch_BFGS(x0s, obj, max_iter=self.BFGS_max_iter//(self.n_repos+1), line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult, threshhold=0.03, progress=lambda x: demo_progress_updater([x[0]/(self.n_repos+1) + i/(self.n_repos+1),x[1]],progress,desc='Stage 2: '))
            x0s = x
        
        x,f = Batch_BFGS(x0s, obj, max_iter=self.BFGS_max_iter - self.n_repos* self.BFGS_max_iter//(self.n_repos+1), line_search_max_iter=self.BFGS_lineserach_max_iter, tau=self.BFGS_line_search_mult, threshhold=0.03, progress=lambda x: demo_progress_updater([x[0]/(self.n_repos+1) + self.n_repos/(self.n_repos+1),x[1]],progress,desc='Stage 2: '))
        
        best_idx = f.argmin()

        end_time = time.time()
        
        if partial:
            target_uni = uniformize(target_curve_copy.unsqueeze(0),self.optim_timesteps)[0]
            
            tr,sc,an = find_transforms(uniformize(target_curve_.unsqueeze(0),self.optim_timesteps),target_uni, batch_ordered_distance)
            transformed_curve = apply_transforms(target_uni.unsqueeze(0),tr,sc,-an)[0]
            end_point = target_curve_[size-1]
            matched_point_idx = torch.argmin(torch.linalg.norm(transformed_curve-end_point,dim=-1))
            
            sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],torch.linspace(0,torch.pi*2,self.optim_timesteps).to(self.device))
            tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
            best_matches = sol[np.arange(sol.shape[0]),tid]
            original_match = best_matches.clone()
            best_matches = uniformize(best_matches,self.optim_timesteps)
            
            tr,sc,an = find_transforms(best_matches,target_uni, batch_ordered_distance)
            tiled_curves = uniformize(target_uni[:matched_point_idx].unsqueeze(0),self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
            transformed_curve = tiled_curves[0].detach().cpu().numpy()
            
            best_matches = get_partial_matches(best_matches,tiled_curves[0],batch_ordered_distance)
            
            CD = batch_chamfer_distance(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            OD = ordered_objective_batch(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            
            st_id, en_id = get_partial_index(original_match,tiled_curves[0],batch_ordered_distance)
            
            st_theta = torch.linspace(0,2*np.pi,self.optim_timesteps).to(self.device)[st_id].squeeze().cpu().numpy()
            en_theta = torch.linspace(0,2*np.pi,self.optim_timesteps).to(self.device)[en_id].squeeze().cpu().numpy()
            
            st_theta[st_theta>en_theta] = st_theta[st_theta>en_theta] - 2*np.pi
            
        else:
            sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],torch.linspace(0,torch.pi*2,self.optim_timesteps).to(self.device))
            tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
            best_matches = sol[np.arange(sol.shape[0]),tid]
            best_matches = uniformize(best_matches,self.optim_timesteps)
            target_uni = uniformize(target_curve_copy.unsqueeze(0),self.optim_timesteps)[0]
            
            tr,sc,an = find_transforms(best_matches,target_uni, batch_ordered_distance)
            tiled_curves = uniformize(target_curve_copy.unsqueeze(0),self.optim_timesteps)
            tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
            transformed_curve = tiled_curves[0].detach().cpu().numpy()
            
            CD = batch_chamfer_distance(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            OD = ordered_objective_batch(best_matches/sc.unsqueeze(-1).unsqueeze(-1),tiled_curves/sc.unsqueeze(-1).unsqueeze(-1))
            
            st_theta = 0.
            en_theta = np.pi*2
        
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        ax = draw_mechanism(As[best_idx].cpu().numpy(),x[best_idx].cpu().numpy(),np.where(node_types[best_idx].cpu().numpy())[0],[0,1],highlight=tid[0].item(),solve=True, thetas=np.linspace(st_theta,en_theta,self.optim_timesteps),ax=ax)
        ax.plot(transformed_curve[:,0], transformed_curve[:,1], color="indigo", alpha=0.7, linewidth=2)
        
        A = As[best_idx].cpu().numpy()
        x = x[best_idx].cpu().numpy()
        node_types = node_types[best_idx].cpu().numpy()
        
        n_joints = (A.sum(-1)>0).sum()
        
        A = A[:n_joints,:][:,:n_joints]
        x = x[:n_joints]
        node_types = node_types[:n_joints]
        
        transformation = [tr.cpu().numpy(),sc.cpu().numpy(),an.cpu().numpy()]
        start_theta = st_theta
        end_theta = en_theta
        performance = [CD.item()*og_scale,OD.item()*(og_scale**2)]
        torch.cuda.empty_cache()
        return fig, [[A,x,node_types, start_theta, end_theta, transformation], performance, transformed_curve], gr.update(value = {"Progress":1.0})
        
def get_partial_matches(curves, target_curve, objective_fn):
    start_point = target_curve[0]
    end_point = target_curve[-1]
    
    start_match_idx = torch.linalg.norm(start_point-curves,dim=-1).argmin(-1)
    end_match_idx = torch.linalg.norm(end_point-curves,dim=-1).argmin(-1)
    
    test_target = torch.cat([target_curve.unsqueeze(0),target_curve.unsqueeze(0)],0)
    
    for i in range(curves.shape[0]):
        if start_match_idx[i] < end_match_idx[i]:
            partial_1 = uniformize(curves[i][start_match_idx[i]:end_match_idx[i]+1].unsqueeze(0),target_curve.shape[0])[0]
            partial_2 = uniformize(torch.cat([curves[i][end_match_idx[i]:],curves[i][:start_match_idx[i]+1]],0).unsqueeze(0),target_curve.shape[0])[0]
            partials = torch.cat([partial_1.unsqueeze(0),partial_2.unsqueeze(0)],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            idx = f.argmin()
            curves[i] = uniformize(partials[idx].squeeze().unsqueeze(0),target_curve.shape[0])[0]
        else:
            partial_1 = uniformize(curves[i][end_match_idx[i]:start_match_idx[i]+1].unsqueeze(0),target_curve.shape[0])[0]
            partial_2 = uniformize(torch.cat([curves[i][start_match_idx[i]:],curves[i][:end_match_idx[i]+1]],0).unsqueeze(0),target_curve.shape[0])[0]
            partials = torch.cat([partial_1.unsqueeze(0),partial_2.unsqueeze(0)],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            idx = f.argmin()
            curves[i] = uniformize(partials[idx].squeeze().unsqueeze(0),target_curve.shape[0])[0]
            
    return curves

def get_partial_index(curves, target_curve, objective_fn):
    start_point = target_curve[0]
    end_point = target_curve[-1]
    
    start_match_idx = torch.linalg.norm(start_point-curves,dim=-1).argmin(-1)
    end_match_idx = torch.linalg.norm(end_point-curves,dim=-1).argmin(-1)
    
    
    actual_start = start_match_idx.clone()
    actual_end = end_match_idx.clone()
    
    test_target = torch.cat([target_curve.unsqueeze(0),target_curve.unsqueeze(0)],0)
    
    for i in range(curves.shape[0]):
        if start_match_idx[i] < end_match_idx[i]:
            partial_1 = uniformize(curves[i][start_match_idx[i]:end_match_idx[i]+1].unsqueeze(0),target_curve.shape[0])[0]
            partial_2 = uniformize(torch.cat([curves[i][end_match_idx[i]:],curves[i][:start_match_idx[i]+1]],0).unsqueeze(0),target_curve.shape[0])[0]
            partials = torch.cat([partial_1.unsqueeze(0),partial_2.unsqueeze(0)],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            idx = f.argmin()
            if idx == 1:
                actual_start[i], actual_end[i] = end_match_idx[i], start_match_idx[i]
        else:
            partial_1 = uniformize(curves[i][end_match_idx[i]:start_match_idx[i]+1].unsqueeze(0),target_curve.shape[0])[0]
            partial_2 = uniformize(torch.cat([curves[i][start_match_idx[i]:],curves[i][:end_match_idx[i]+1]],0).unsqueeze(0),target_curve.shape[0])[0]
            partials = torch.cat([partial_1.unsqueeze(0),partial_2.unsqueeze(0)],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            idx = f.argmin()
            
            if idx == 0:
                actual_start[i], actual_end[i] = end_match_idx[i], start_match_idx[i]
    
    return actual_start, actual_end

def get_partial_matches_oto(curves, target_curves, objective_fn):
    
    for i in range(curves.shape[0]):
        start_point = target_curves[i][0]
        end_point = target_curves[i][-1]
        
        start_match_idx = torch.linalg.norm(start_point-curves[i:i+1],dim=-1).argmin(-1).squeeze()
        end_match_idx = torch.linalg.norm(end_point-curves[i:i+1],dim=-1).argmin(-1).squeeze()
        
        test_target = torch.cat([target_curves[i].unsqueeze(0),target_curves[i].unsqueeze(0)],0)
        
        if start_match_idx < end_match_idx:
            partial_1 = uniformize(curves[i][start_match_idx:end_match_idx+1].unsqueeze(0),curves.shape[1])[0]
            partial_2 = uniformize(torch.cat([curves[i][end_match_idx:],curves[i][:start_match_idx+1]],0).unsqueeze(0),curves.shape[1])[0]
            partials = torch.cat([partial_1.unsqueeze(0),partial_2.unsqueeze(0)],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            # idx = f.argmin()
            if f[0] < f[1]:
                curves[i] = uniformize(curves[i][start_match_idx:end_match_idx+1].unsqueeze(0),curves.shape[1])[0]
            else:
                curves[i] = uniformize(torch.cat([curves[i][end_match_idx:],curves[i][:start_match_idx+1]],0).unsqueeze(0),curves.shape[1])[0]
            # curves[i] = uniformize(partials[idx].squeeze().unsqueeze(0),curves.shape[1])[0]
        else:
            partial_1 = uniformize(curves[i][end_match_idx:start_match_idx+1].unsqueeze(0),curves.shape[1])[0]
            partial_2 = uniformize(torch.cat([curves[i][start_match_idx:],curves[i][:end_match_idx+1]],0).unsqueeze(0),curves.shape[1])[0]
            partials = torch.cat([partial_1.unsqueeze(0),partial_2.unsqueeze(0)],0)
            
            f = objective_fn(partials,test_target).reshape(-1)
            # idx = f.argmin()
            if f[0] < f[1]:
                curves[i] = uniformize(curves[i][end_match_idx:start_match_idx+1].unsqueeze(0),curves.shape[1])[0]
            else:
                curves[i] = uniformize(torch.cat([curves[i][start_match_idx:],curves[i][:end_match_idx+1]],0).unsqueeze(0),curves.shape[1])[0]
            # curves[i] = uniformize(partials[idx].squeeze().unsqueeze(0),curves.shape[1])[0]
            
    return curves