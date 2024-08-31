import os
import uuid
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--cuda_device", type=str, default='0', help="Cuda devices to use. Default is 0")
argparser.add_argument('--checkpoint_folder', type=str, default='./Checkpoints/', help='The folder to store the checkpoint')
argparser.add_argument('--checkpoint_name', type=str, default='checkpoint.LInK', help='The name of the checkpoint file')
argparser.add_argument('--data_folder', type=str, default='./Data/', help='The folder to store the data')
argparser.add_argument('--embedding_folder', type=str, default='./Embeddings/', help='The folder to store the embeddings')
argparser.add_argument('--test_curves', type=str, default='./TestData/alphabet.npy', help='The path to a npy file containing the test curves you want to run LInK on. Default is ./TestData/alphabet.npy')
argparser.add_argument("--save_name", type=str, default='results.npy', help="The name of the file to save the synthesized mechanisms. Default is results.npy")
argparser.add_argument("--vis_folder", type=str, default='./Vis/', help="The folder to save the visualization of the synthesized mechanisms. Default is ./Vis/ only used if vis_candidates is True.")
argparser.add_argument("--vis_candidates", type=bool, default=True, help="Whether to visualize the retrieved candidates. Default is True.")

argparser.add_argument('--max_joint', type=int, default=20, help='The maximum number of joints to use in the synthesis. Default is 20')
argparser.add_argument('--optim_timesteps', type=int, default=2000, help='The number of timesteps to use in the solver. Default is 2000')
argparser.add_argument('--top_n', type=int, default=300, help='The number of paths to keep in the first level of the optimization. Default is 300')
argparser.add_argument('--init_optim_iters', type=int, default=10, help='The number of iterations to run the first level of optimization. Default is 10')
argparser.add_argument('--top_n_level2', type=int, default=30, help='The number of paths to keep in the second level of the optimization. Default is 30')
argparser.add_argument('--CD_weight', type=float, default=1.0, help='The weight of the chamfer distance in the optimization. Default is 1.0')
argparser.add_argument('--OD_weight', type=float, default=0.25, help='The weight of the ordered distance in the optimization. Default is 0.25')
argparser.add_argument('--BFGS_max_iter', type=int, default=150, help='The maximum number of iterations to run the BFGS optimization. Default is 150')
argparser.add_argument('--n_repos', type=int, default=1, help='The number of repositioning to use in the optimization. Default is 1')
argparser.add_argument('--BFGS_lineserach_max_iter', type=int, default=10, help='The maximum number of line search iterations to use in the optimization. Default is 10')
argparser.add_argument('--BFGS_line_search_mult', type=float, default=0.5, help='The multiplier to use in the line search. Default is 0.5')
argparser.add_argument('--curve_size', type=int, default=200, help='The number of points to uniformize the curves. Default is 200')
argparser.add_argument('--smoothing', type=bool, default=True, help='Whether to smooth the curves. Default is True. Recommended to keep it True only for hand drawn curves or curves with high noise')
argparser.add_argument('--n_freq', type=int, default=7, help='The number of frequencies to use in the smoothing. Default is 7')


args = argparser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

from LInK.OptimJax import PathSynthesis
from pathlib import Path

import numpy as np
import pickle
import torch
import jax

# turn off gradient computation
torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the checkpoint
if not os.path.exists(args.checkpoint_folder) or not os.path.exists(os.path.join(args.checkpoint_folder, args.checkpoint_name)):
    raise ValueError('The checkpoint file does not exist please run Download.py to download the checkpoints or provide the correct path.')

# load the model
if device == 'cpu':
    with open(os.path.join(args.checkpoint_folder, args.checkpoint_name), 'rb') as f:
        Trainer = pickle.load(f)
else:
      with open(os.path.join(args.checkpoint_folder, args.checkpoint_name), 'rb') as f:
        Trainer = pickle.load(f)
        
Trainer.model_base = Trainer.model_base.to('cpu')
Trainer.model_mechanism = Trainer.model_mechanism.to('cpu')
Trainer.model_input.compile()

# load data
if not os.path.exists(args.data_folder) or not os.path.exists(os.path.join(args.data_folder, 'target_curves.npy')) or not os.path.exists(os.path.join(args.data_folder, 'connectivity.npy')) or not os.path.exists(os.path.join(args.data_folder, 'x0.npy')) or not os.path.exists(os.path.join(args.data_folder, 'node_types.npy')):
    raise ValueError('All or some of the data does not exist please run Download.py to download the data or provide the correct path.')

if not os.path.exists(args.embedding_folder) or not os.path.exists(os.path.join(args.embedding_folder, 'embeddings.npy')):
    raise ValueError('The embedding file does not exist please run Download.py to download the embedding file or run Precompute.py to recompute them or provide the correct path.')

if not os.path.exists(args.test_curves):
    raise ValueError('The test curves file does not exist please provide the correct path.')

if not os.path.exists(args.vis_folder):
    os.mkdir(args.vis_folder)

emb  = np.load(os.path.join(args.embedding_folder, 'embeddings.npy'))[0:2000000]
# emb = torch.tensor(emb).float().to(device)
emb = jax.numpy.array(emb, dtype=jax.numpy.float32)
As = np.load(os.path.join(args.data_folder, 'connectivity.npy'))[0:2000000]
x0s = np.load(os.path.join(args.data_folder, 'x0.npy'))[0:2000000]
node_types = np.load(os.path.join(args.data_folder, 'node_types.npy'))[0:2000000]
curves = np.load(os.path.join(args.data_folder, 'target_curves.npy'))[0:2000000]
sizes = (As.sum(-1)>0).sum(-1)

torch.cuda.empty_cache()

test_curves = np.load(args.test_curves,allow_pickle=True)
synthsizer = PathSynthesis(Trainer, curves, As, x0s, node_types, emb, BFGS_max_iter=args.BFGS_max_iter, n_repos=args.n_repos, BFGS_lineserach_max_iter=args.BFGS_lineserach_max_iter, BFGS_line_search_mult=args.BFGS_line_search_mult, curve_size=args.curve_size, smoothing=args.smoothing, n_freq=args.n_freq, CD_weight=args.CD_weight, OD_weight=args.OD_weight, top_n=args.top_n, top_n_level2=args.top_n_level2, init_optim_iters=args.init_optim_iters, optim_timesteps=args.optim_timesteps)

results = []

for i in range(len(test_curves)):
    print('Synthesizing curve:', i)
    partial = True
    t = np.linalg.norm(test_curves[i][0] - test_curves[i][1])
    e = np.linalg.norm(test_curves[i][0] - test_curves[i][-1])
    if e/t <= 1.1:
        partial = False
    synth_curves = synthsizer.synthesize(test_curves[i], verbose=True, visualize=args.vis_candidates, save_figs=os.path.join(args.vis_folder, 'curve_' + str(i)), max_size=args.max_joint, partial=partial)
    results.append(synth_curves)

np.save(args.save_name, results)