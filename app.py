import os
import uuid
import pyvista as pv
import argparse
pv.start_xvfb()

argparser = argparse.ArgumentParser()
argparser.add_argument("--port", type=int, default=1238, help="Port number for the local server")
argparser.add_argument("--cuda_device", type=str, default='0', help="Cuda devices to use. Default is 0")
argparser.add_argument("--static_folder", type=str, default='static', help="Folder to store static files")
args = argparser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import gradio as gr
from LInK.demo import draw_html, draw_script, css
from LInK.Solver import solve_rev_vectorized_batch_CPU
from LInK.CAD import get_layers, plot_config_3D, plot_3D_animation, create_3d_html
from LInK.Optim import PathSynthesis
from pathlib import Path

import numpy as np
import pickle
import torch


# turn off gradient computation
torch.set_grad_enabled(False)

# check if the static folder exists
if not Path(args.static_folder).exists():
    os.mkdir(args.static_folder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
if device == 'cpu':
    with open('./Checkpoints/checkpointCPU.LInK', 'rb') as f:
        Trainer = pickle.load(f)
else:
    with open('./Checkpoints/checkpoint.LInK', 'rb') as f:
        Trainer = pickle.load(f)
        
Trainer.model_base = Trainer.model_base.to('cpu')
Trainer.model_mechanism = Trainer.model_mechanism.to('cpu')
  
# load data
emb  = np.load('./Embeddings/embeddings.npy')[0:2000000]
emb = torch.tensor(emb).float().to(device)
As = np.load('./Data/connectivity.npy')[0:2000000]
x0s = np.load('./Data/x0.npy')[0:2000000]
node_types = np.load('./Data/node_types.npy')[0:2000000]
curves = np.load('./Data/target_curves.npy')[0:2000000]
sizes = (As.sum(-1)>0).sum(-1)

torch.cuda.empty_cache()

def create_synthesizer(n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter):
    mask = (sizes<=maximum_joint_count)
    synthesizer = PathSynthesis(Trainer, curves[mask], As[mask], x0s[mask], node_types[mask], emb[mask], BFGS_max_iter=BFGS_max_iter, n_freq=n_freq, optim_timesteps=time_steps, top_n=top_n, init_optim_iters=init_optim_iters, top_n_level2=top_n_level2)
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

gr.set_static_paths(paths=[Path('./static')])

with gr.Blocks(css=css, js=draw_script) as block:
    
    syth  = gr.State()
    state = gr.State()
    dictS = gr.State(False)
    
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
            time_steps = gr.Slider(minimum = 200 , maximum = 5000, value=2000, step=1, label="Number of Simulation Time Steps", interactive=True)
            top_n = gr.Slider(minimum = 50 , maximum = 1000, value=300, step=1, label="Top N Candidates To Start With", interactive=True)
            init_optim_iters = gr.Slider(minimum = 10 , maximum = 50, value=20, step=1, label="Initial Optimization Iterations On All Candidates", interactive=True)
            top_n_level2 = gr.Slider(minimum = 10 , maximum = 100, value=30, step=1, label="Top N Candidates For Final Optimization", interactive=True)
            BFGS_max_iter = gr.Slider(minimum = 50 , maximum = 500, value=200, step=1, label="Iterations For Final Optimization", interactive=True)
        
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
    
    event1 = btn_submit.click(lambda: [None]*4 + [gr.update(interactive=False)]*8, outputs=[candidate_plt,mechanism_plot,og_plt,smooth_plt,btn_submit, n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter], concurrency_limit=1)
    event2 = event1.then(create_synthesizer, inputs=[n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter], outputs=[syth], concurrency_limit=1)
    event3 = event2.then(lambda s,x,p: s.demo_sythesize_step_1(np.array([eval(i) for i in x.split(',')]).reshape([-1,2]) * [[1,-1]],partial=p), inputs=[syth,canvas,partial],js="(s,x,p) => [s,path.toString(),p]",outputs=[state,og_plt,smooth_plt], concurrency_limit=1)
    event4 = event3.then(lambda sy,s: sy.demo_sythesize_step_2(s), inputs=[syth,state], outputs=[state,candidate_plt], concurrency_limit=1)
    event5 = event4.then(lambda sy,s: sy.demo_sythesize_step_3(s,progress=gr.Progress()), inputs=[syth,state], outputs=[mechanism_plot,state,progl], concurrency_limit=1)
    event6 = event5.then(make_cad, inputs=[state,partial], outputs=[plot_3d], concurrency_limit=1)
    event8 = event6.then(lambda: [gr.update(interactive=True)]*8, outputs=[btn_submit, n_freq, maximum_joint_count, time_steps, top_n, init_optim_iters, top_n_level2, BFGS_max_iter], concurrency_limit=1)
    block.load()
    
    clr_btn.click(lambda x: x, js='document.getElementById("sketch").innerHTML = ""')
    
block.launch(server_port=1238,share=False,max_threads=200,inline=False)