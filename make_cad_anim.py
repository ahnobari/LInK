from LInK.demo import draw_html, draw_script
from LInK.Solver import solve_rev_vectorized_batch_CPU
from LInK.CAD import get_layers, plot_config_3D, plot_3D_animation

import numpy as np
import pickle
import torch

from LInK.Optim import PathSynthesis
# turn off gradient computation
torch.set_grad_enabled(False)

import uuid

import pyvista as pv
pv.start_xvfb()

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--input', type=str, default='input.pkl')

args = argparser.parse_args()

input_path = args.input

with open(input_path, 'rb') as f:
    synth_out = pickle.load(f)

f_name = input_path.replace('.pkl','')


A_M, x0_M, node_types_M, start_theta_M, end_theta_M, tr_M = synth_out[0]
sol_m = solve_rev_vectorized_batch_CPU(A_M[np.newaxis], x0_M[np.newaxis],node_types_M[np.newaxis],np.linspace(start_theta_M, end_theta_M, 200))[0]
x0_M_ = sol_m[:,0,:]
center_M = synth_out[-1].mean(0)
scale_M = np.linalg.norm(sol_m[-1] - center_M,axis=-1).max()
x0_M_ = (x0_M_ - center_M)/scale_M

l_M,s_M,r_M = get_layers(A_M, x0_M, node_types_M,sol_m,verbose=False)

theta_anim = np.linspace(start_theta_M, end_theta_M, 30)
sol_m_ = solve_rev_vectorized_batch_CPU(A_M[np.newaxis], x0_M_[np.newaxis]*0.3,node_types_M[np.newaxis],theta_anim)[0]
# sol_m_ = np.concatenate([sol_m_,sol_m_[:,::-1,:]],axis=1)
plot_3D_animation(A_M, x0_M_*0.3, node_types_M, l_M.astype(int), [sol_m_[-1]],h_joints=[A_M.shape[0]-1],sol=sol_m_.transpose(1,0,2),file=f'{f_name}.gif')