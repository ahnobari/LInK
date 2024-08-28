import os
import uuid
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--port", type=int, default=1238, help="Port number for the local server")
argparser.add_argument("--cuda_device", type=str, default='0', help="Cuda devices to use. Default is 0")
argparser.add_argument("--static_folder", type=str, default='static', help="Folder to store static files")
argparser.add_argument('--checkpoint_folder', type=str, default='./Checkpoints/', help='The folder to store the checkpoint')
argparser.add_argument('--checkpoint_name', type=str, default='checkpoint.LInK', help='The name of the checkpoint file')
argparser.add_argument('--data_folder', type=str, default='./Data/', help='The folder to store the data')
argparser.add_argument('--embedding_folder', type=str, default='./Embeddings/', help='The folder to store the embeddings')
args = argparser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import gradio as gr
from LInK.demo import draw_html, draw_script, css
from LInK.Solver import solve_rev_vectorized_batch_CPU
from LInK.CAD import get_layers, create_3d_html
from LInK.OptimJax import PathSynthesis
from pathlib import Path
import jax

import numpy as np
import pickle
import torch


# turn off gradient computation
torch.set_grad_enabled(False)

# check if the static folder exists
if not Path(args.static_folder).exists():
    os.mkdir(args.static_folder)

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

emb  = np.load(os.path.join(args.embedding_folder, 'embeddings.npy'))[0:2000000]
# emb = torch.tensor(emb).float().to(device)
emb = jax.numpy.array(emb, dtype=jax.numpy.float32)
As = np.load(os.path.join(args.data_folder, 'connectivity.npy'))[0:2000000]
x0s = np.load(os.path.join(args.data_folder, 'x0.npy'))[0:2000000]
node_types = np.load(os.path.join(args.data_folder, 'node_types.npy'))[0:2000000]
curves = np.load(os.path.join(args.data_folder, 'target_curves.npy'))[0:2000000]
sizes = (As.sum(-1)>0).sum(-1)

torch.cuda.empty_cache()

def create_synthesizer(n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter):
    # mask = (sizes<=maximum_joint_count)
    synthesizer = PathSynthesis(Trainer, curves, As, x0s, node_types, emb, BFGS_max_iter=BFGS_max_iter, n_freq=n_freq, optim_timesteps=time_steps, top_n=top_n, init_optim_iters=init_optim_iters, top_n_level2=top_n_level2)
    return synthesizer
    
def make_cad(synth_out, partial, progress=gr.Progress(track_tqdm=True)):
    
    progress(0, desc="Generating 3D Model ...")
    
    f_name = str(uuid.uuid4())
    
    A_M, x0_M, node_types_M, start_theta_M, end_theta_M, tr_M = synth_out[0]
    
    sol_m = solve_rev_vectorized_batch_CPU(A_M[np.newaxis], x0_M[np.newaxis],node_types_M[np.newaxis],np.linspace(start_theta_M, end_theta_M, 200))[0]
    z,status = get_layers(A_M, x0_M, node_types_M,sol_m)
    
    if partial:
        sol_m = np.concatenate([sol_m, sol_m[:,::-1,:]], axis=1)
    
    create_3d_html(A_M, x0_M, node_types_M, z, sol_m, template_path = f'./{args.static_folder}/animation.html', save_path=f'./{args.static_folder}/{f_name}.html')
    
    return gr.HTML(f'<iframe width="100%" height="800px" src="file={args.static_folder}/{f_name}.html"></iframe>',label="3D Plot",elem_classes="plot3d")

gr.set_static_paths(paths=[Path(f'./{args.static_folder}')])


with gr.Blocks(css=css, js=draw_script) as block:
    
    syth  = gr.State()
    state = gr.State()
    dictS = gr.State(False)
    
    with gr.Row():
        intro = gr.Markdown('''
        # LInK: Learning Joint Representations of Design and Performance Spaces through Contrastive Learning for Mechanism Synthesis
        LInK is a novel framework that integrates contrastive learning of performance and design space with optimization techniques for solving complex inverse problems in engineering design with discrete and continuous variables. We focus on the path synthesis problem for planar linkage mechanisms in this application.

        [<img src="https://github.com/user-attachments/assets/7e6790c7-fbbe-4658-96f7-25aad304c59d" style="width:100%; max-width:1000px;"/>](https://ahn1376-linkalphabetdemo.hf.space/)

        If you want to see the alphabet solutions, please visit look at the simple demo we have on huggingface:
        [Simple Fun Alphabet Demo](https://ahn1376-linkalphabetdemo.hf.space/)

        Code & Data: [GitHub](https://github.com/ahnobari/LInK/)

        Paper (Currently Under Review): [arXiv](https://arxiv.org/abs/2405.20592)

        Below you can draw a curve and synthesize a mechanism that can trace the curve. You can also adjust the algorithm parameters to see how it affects the solution.
        ''', elem_classes="intro")

    with gr.Row():

        with gr.Column(min_width=350,scale=2):
            canvas = gr.HTML(draw_html)
            clr_btn = gr.Button("Clear",elem_classes="clr_btn")
            
            btn_submit = gr.Button("Perform Path Synthesis",variant='primary',elem_classes="clr_btn")
            
            # checkbox
            partial = gr.Checkbox(label="Partial Curve", value=False, elem_id="partial")

        with gr.Column(min_width=250,scale=1,visible=True):
            gr.HTML("<h2>Algorithm Parameters</h2>")
            
            n_freq = gr.Slider(minimum = 3 , maximum = 50, value=7, step=1, label="Number of Frequenceies For smoothing", interactive=True)
            maximum_joint_count = gr.Slider(minimum = 6 , maximum = 20, value=14, step=1, label="Maximum Joint Count", interactive=True)
            time_steps = gr.Slider(minimum = 1000 , maximum = 3000, value=2000, step=500, label="Number of Simulation Time Steps", interactive=True, visible=False)
            top_n = gr.Slider(minimum = 50 , maximum = 1000, value=300, step=50, label="Top N Candidates To Start With", interactive=True, visible=False)
            init_optim_iters = gr.Slider(minimum = 10 , maximum = 50, value=20, step=10, label="Initial Optimization Iterations On All Candidates", interactive=True)
            top_n_level2 = gr.Slider(minimum = 10 , maximum = 100, value=40, step=10, label="Top N Candidates For Final Optimization", interactive=True, visible=False)
            BFGS_max_iter = gr.Slider(minimum = 50 , maximum = 1000, value=200, step=50, label="Iterations For Final Optimization", interactive=True)
        
    with gr.Row():
        with gr.Row():
            with gr.Column(min_width=250,scale=1,visible=True):
                gr.HTML('<h2>Algorithm Outputs</h2>')
                progl = gr.Label({"Progress": 0}, elem_classes="prog",num_top_classes=1)
                
    with gr.Row():
        with gr.Column(min_width=250,visible=True):
            og_plt = gr.Plot(label="Original Input",elem_classes="plotpad")
        with gr.Column(min_width=250,visible=True):    
            smooth_plt = gr.Plot(label="Smoothed Drawing",elem_classes="plotpad")
                
    with gr.Row():
        candidate_plt = gr.Plot(label="Initial Candidates",elem_classes="plotpad")
    
    with gr.Row():
        mechanism_plot = gr.Plot(label="Solution",elem_classes="plotpad")
        

    with gr.Row():
        plot_3d = gr.HTML('<iframe width="100%" height="800px" src="file=static/filler.html"></iframe>',label="3D Plot",elem_classes="plot3d")
    
    event1 = btn_submit.click(lambda: [None]*4 + [gr.update(interactive=False)]*8, outputs=[candidate_plt,mechanism_plot,og_plt,smooth_plt,btn_submit, n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter], concurrency_limit=10)
    event2 = event1.then(create_synthesizer, inputs=[n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter], outputs=[syth], concurrency_limit=10)
    event3 = event2.then(lambda s,x,p: s.demo_sythesize_step_1(np.array([eval(i) for i in x.split(',')]).reshape([-1,2]) * [[1,-1]],partial=p), inputs=[syth,canvas,partial],js="(s,x,p) => [s,path.toString(),p]",outputs=[state,og_plt,smooth_plt], concurrency_limit=10)
    event4 = event3.then(lambda sy,s,mj: sy.demo_sythesize_step_2(s,max_size=mj), inputs=[syth,state,maximum_joint_count], outputs=[state,candidate_plt], concurrency_limit=10)
    event5 = event4.then(lambda sy,s: sy.demo_sythesize_step_3(s,progress=gr.Progress()), inputs=[syth,state], outputs=[mechanism_plot,state,progl], concurrency_limit=10)
    event6 = event5.then(make_cad, inputs=[state,partial], outputs=[plot_3d], concurrency_limit=10)
    event8 = event6.then(lambda: [gr.update(interactive=True)]*8, outputs=[btn_submit, n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter], concurrency_limit=10)
    block.load()
    
    clr_btn.click(lambda x: x, js='document.getElementById("sketch").innerHTML = ""')
    
block.launch(root_path='/linkage', server_name='10.80.6.47', server_port=args.port,share=True,max_threads=200,inline=False,)