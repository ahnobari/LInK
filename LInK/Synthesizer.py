from .OptimJax import *
from .CAD import get_layers

def Synthesize(models,
               As, 
               x0s, 
               node_types, 
               target_curve, 
               curves, 
               sizes, 
               precomputed_emb, 
               max_size = 20, 
               curve_size = 200, 
               n_freq=7, 
               smoothing=True, 
               partial=False, 
               top_n=100, 
               optim_timesteps=2000, 
               OD_weight=0.25, 
               CD_weight=1.0, 
               init_optim_iters = 20,
               BFGS_lineserach_max_iter = 10,
               BFGS_line_search_mult = 0.5,
               top_n_level2 = 30,
               n_repos = 1,
               BFGS_max_iter = 200,
               manufacturable = False,
               device=None,
               verbose=False):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    start_time = time.time()
    
    target_curve = preprocess_curves(target_curve[None],curve_size)[0]
    
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
        
        angles = jax.numpy.linspace(start_angle,end_angle,curve_size)
        ellipse = jax.numpy.stack([center[0] + a*jax.numpy.cos(angles),center[1] + b*jax.numpy.sin(angles)],1)
        
        angles = jax.numpy.linspace(start_angle+2*np.pi,end_angle,curve_size)
        ellipse_2 = jax.numpy.stack([center[0] + a*jax.numpy.cos(angles),center[1] + b*jax.numpy.sin(angles)],1)
        
        #ellipse 1 length
        l_1 = jax.numpy.linalg.norm(ellipse - target_curve.mean(0) , axis=-1).sum()
        #ellipse 2 length
        l_2 = jax.numpy.linalg.norm(ellipse_2 - target_curve.mean(0),axis=-1).sum()
        
        if l_1 > l_2:
            target_curve = jax.numpy.concatenate([target_curve,ellipse],0)
        else:
            target_curve = jax.numpy.concatenate([target_curve,ellipse_2],0)
    
    target_curve_copy = preprocess_curves(target_curve[None], curve_size)[0]
    target_curve_ = jax.numpy.copy(target_curve)
    
    if smoothing:
        target_curve = smooth_hand_drawn_curves(target_curve[None], n=curve_size, n_freq=n_freq)[0]
    else:
        target_curve = preprocess_curves(target_curve[None],curve_size)[0]
    
    if partial:
        # target_curve_copy_ = preprocess_curves(target_curve_[:size][None], curve_size)[0]
        tr,sc,an = find_transforms(uniformize(target_curve_[None],curve_size),target_curve, )
        transformed_curve = apply_transforms(target_curve[None],tr,sc,-an)[0]
        end_point = target_curve_[size-1]
        matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve-end_point,axis=-1))
        target_curve = preprocess_curves(target_curve[:matched_point_idx+1][None],curve_size)[0]
        
        target_uni = jax.numpy.copy(target_curve_copy)
        
        tr,sc,an = find_transforms(uniformize(target_curve_[None],curve_size),target_uni, )
        transformed_curve = apply_transforms(target_uni[None],tr,sc,-an)[0]
        end_point = target_curve_[size-1]
        matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve-end_point,axis=-1))
        target_curve_copy_ = uniformize(target_curve_copy[:matched_point_idx+1][None],curve_size)[0]
    else:
        target_curve_copy_ = jax.numpy.copy(target_curve_copy)
    
    # fig1 = plt.figure(figsize=(5,5))
    # if partial:
    #     plt.plot(target_curve_[:size][:,0],target_curve_[:size][:,1],color="indigo")
    # else:
    #     plt.plot(target_curve_copy[:,0],target_curve_copy[:,1],color="indigo")
    # plt.axis('equal')
    # plt.axis('off')
    # plt.title('Original Curve')
    
    # fig2 = plt.figure(figsize=(5,5))
    # plt.plot(target_curve[:,0],target_curve[:,1],color="indigo")
    # plt.axis('equal')
    # plt.axis('off')
    # plt.title('Preprocessed Curve')

    input_tensor = target_curve[None]
    batch_padd = preprocess_curves(curves[np.random.choice(curves.shape[0],255)])

    input_tensor = torch.tensor(np.concatenate([input_tensor,batch_padd],0)).float().to(device)
    with torch.cuda.amp.autocast():
        target_emb = models.compute_embeddings_input(input_tensor, 1000)[0]
    # target_emb = torch.tensor(target_emb).float().to(device)
    
    ids =np.where(sizes <= max_size)[0]
    idxs, sim = cosine_search_jax(target_emb, precomputed_emb, ids=ids)
    # idxs = idxs.detach().cpu().numpy()
    
    #max batch size is 250
    tr,sc,an = [],[],[]
    for i in range(int(np.ceil(top_n*5/250))):
        tr_,sc_,an_ = find_transforms(curves[idxs[i*250:(i+1)*250]],target_curve_copy, )
        tr.append(tr_)
        sc.append(sc_)
        an.append(an_)
    tr = jax.numpy.concatenate(tr,0)
    sc = jax.numpy.concatenate(sc,0)
    an = jax.numpy.concatenate(an,0)
    tiled_curves = target_curve_copy[None].repeat(top_n*5,0)
    tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
    CD = batch_ordered_distance(tiled_curves/sc[:,None,None],curves[idxs[:top_n*5]]/sc[:,None,None])
    
    #get best matches index
    tid = jax.numpy.argsort(CD)[:top_n]
    
    # grid_size = int(np.ceil(top_n**0.5))
    # fig, axs = plt.subplots(grid_size,grid_size,figsize=(10,10))
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         if i*grid_size+j < top_n:
    #             axs[i,j].plot(curves[idxs[tid][i*grid_size+j]][:,0],curves[idxs[tid][i*grid_size+j]][:,1], color='indigo')
    #             axs[i,j].plot(tiled_curves[tid][i*grid_size+j][:,0],tiled_curves[tid][i*grid_size+j][:,1],color="darkorange",alpha=0.7)
    #         axs[i,j].axis('off')
    #         axs[i,j].axis('equal')

    As = As[idxs[tid]]
    x0s = x0s[idxs[tid]]
    node_types = node_types[idxs[tid]]
    
    if verbose:
        prog = trange(init_optim_iters+BFGS_max_iter)
    else:
        prog = None

    obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types,timesteps=optim_timesteps,OD_weight=OD_weight, CD_weight=CD_weight)
    
    x,f = Batch_BFGS(x0s, obj, max_iter=init_optim_iters, line_search_max_iter=BFGS_lineserach_max_iter, tau=BFGS_line_search_mult, progress=lambda x: progerss_uppdater(x,prog),threshhold=0.0)
    
    # top n level 2
    top_n_2 = f.argsort()[:top_n_level2]
    As = As[top_n_2]
    x0s = x[top_n_2]
    node_types = node_types[top_n_2]
    
    # if partial:
    #     obj = make_batch_optim_obj_partial(target_curve_copy, target_curve_copy_, As, x0s, node_types,timesteps=optim_timesteps,OD_weight=0.25)
    # else:
    obj = make_batch_optim_obj(target_curve_copy, As, x0s, node_types,timesteps=optim_timesteps,OD_weight=OD_weight, CD_weight=CD_weight)

    for i in range(n_repos):
        x,f = Batch_BFGS(x0s, obj, max_iter=BFGS_max_iter//(n_repos+1), line_search_max_iter=BFGS_lineserach_max_iter, tau=BFGS_line_search_mult, threshhold=0., progress=lambda x: progerss_uppdater(x,prog))
        x0s = x
    
    x,f = Batch_BFGS(x0s, obj, max_iter=BFGS_max_iter - n_repos* BFGS_max_iter//(n_repos+1), line_search_max_iter=BFGS_lineserach_max_iter, tau=BFGS_line_search_mult, threshhold=0., progress=lambda x: progerss_uppdater(x,prog))

    end_time = time.time()

    sorted_ids = f.argsort()
    best_idx = sorted_ids[0]

    if manufacturable:
        for i in range(sorted_ids.shape[0]):
            best_idx = sorted_ids[i]
            A_M, x0_M, node_types_M, start_theta_M, end_theta_M = As[best_idx], x0s[best_idx], node_types[best_idx], 0., 2*np.pi
        
            sol_m = solve_rev_vectorized_batch(A_M[np.newaxis], x0_M[np.newaxis],node_types_M[np.newaxis],np.linspace(start_theta_M, end_theta_M, 200))[0]
            z,status = get_layers(A_M, x0_M, node_types_M,sol_m)

            if status:
                break

    if partial:
        target_uni = uniformize(target_curve_copy[None],optim_timesteps)[0]
        
        tr,sc,an = find_transforms(uniformize(target_curve_[None],optim_timesteps),target_uni, )
        transformed_curve = apply_transforms(target_uni[None],tr,sc,-an)[0]
        end_point = target_curve_[size-1]
        matched_point_idx = jax.numpy.argmin(jax.numpy.linalg.norm(transformed_curve-end_point,axis=-1))
        
        sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],jax.numpy.linspace(0,jax.numpy.pi*2,optim_timesteps))
        tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
        best_matches = sol[np.arange(sol.shape[0]),tid]
        original_match = jax.numpy.copy(best_matches)
        best_matches = uniformize(best_matches,optim_timesteps)
        
        tr,sc,an = find_transforms(best_matches,target_uni, )
        tiled_curves = uniformize(target_uni[:matched_point_idx][None],optim_timesteps)
        tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
        transformed_curve = tiled_curves[0]
        
        best_matches = get_partial_matches(best_matches,tiled_curves[0],)
        
        CD = batch_chamfer_distance(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
        OD = ordered_objective_batch(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
        
        st_id, en_id = get_partial_index(original_match,tiled_curves[0],)
        
        st_theta = np.linspace(0,2*np.pi,optim_timesteps)[st_id].squeeze()
        en_theta = np.linspace(0,2*np.pi,optim_timesteps)[en_id].squeeze()
        
        st_theta[st_theta>en_theta] = st_theta[st_theta>en_theta] - 2*np.pi
        
    else:
        sol = solve_rev_vectorized_batch(As[best_idx:best_idx+1],x[best_idx:best_idx+1],node_types[best_idx:best_idx+1],jax.numpy.linspace(0,jax.numpy.pi*2,optim_timesteps))
        tid = (As[best_idx:best_idx+1].sum(-1)>0).sum(-1)-1
        best_matches = sol[np.arange(sol.shape[0]),tid]
        best_matches = uniformize(best_matches,optim_timesteps)
        target_uni = uniformize(target_curve_copy[None],optim_timesteps)[0]
        
        tr,sc,an = find_transforms(best_matches,target_uni, )
        tiled_curves = uniformize(target_curve_copy[None],optim_timesteps)
        tiled_curves = apply_transforms(tiled_curves,tr,sc,-an)
        transformed_curve = tiled_curves[0]
        
        CD = batch_chamfer_distance(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
        OD = ordered_objective_batch(best_matches/sc[:,None,None],tiled_curves/sc[:,None,None])
        
        st_theta = 0.
        en_theta = np.pi*2
    
    n_joints = (As[best_idx].sum(-1)>0).sum()
    # fig, ax = plt.subplots(1,1,figsize=(10,10))
    # ax = draw_mechanism(As[best_idx][:n_joints,:][:,:n_joints],x[best_idx][0:n_joints],np.where(node_types[best_idx][0:n_joints])[0],[0,1],highlight=tid[0].item(),solve=True, thetas=np.linspace(st_theta,en_theta,optim_timesteps),ax=ax)
    # ax.plot(transformed_curve[:,0], transformed_curve[:,1], color="indigo", alpha=0.7, linewidth=2)
    
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
    return A,x,node_types, start_theta, end_theta, transformation, performance, transformed_curve, end_time-start_time