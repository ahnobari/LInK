import os
import uuid
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--cuda_device", type=str, default='0', help="Cuda devices to use. Default is 0")
argparser.add_argument('--checkpoint_folder', type=str, default='./Checkpoints/', help='The folder to store the checkpoint')
argparser.add_argument('--checkpoint_name', type=str, default='checkpoint.LInK', help='The name of the checkpoint file to save, this will be used with _epoch for checkpointing and the main file will always be the latest checkpoint')
argparser.add_argument('--checkpoint_interval', type=int, default=1, help='The epoch interval to save the checkpoint Default is 1')
argparser.add_argument('--data_folder', type=str, default='./Data/', help='The folder to store the data')
argparser.add_argument('--baseline', type=str, default='', help='If set to GCN or GIN the model will train a Naive GCN or GIN model instead of GHOP. Default is GHOP')
argparser.add_argument('--epochs', type=int, default=50, help='The number of epochs to train the model. Default is 50')
argparser.add_argument('--batch_size', type=int, default=256, help='The batch size for training. Default is 256')
argparser.add_argument('--checkpoint_continue', type=str, default='', help='The checkpoint file to continue training from. This should be in the checkpoint folder. Default is empty')
args = argparser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

from pathlib import Path
import numpy as np
import pickle
import torch
from LInK.nn import ContrastiveTrainLoop
from LInK.DataUtils import prep_curves, uniformize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# check the checkpoint folder
if not os.path.exists(args.checkpoint_folder):
    os.mkdir(args.checkpoint_folder)

if args.checkpoint_continue != '':
    if not os.path.exists(os.path.join(args.checkpoint_folder, args.checkpoint_continue)):
        raise ValueError('The checkpoint file does not exist please run Download.py to download the checkpoints or provide the correct path.')
    with open(os.path.join(args.checkpoint_folder, args.checkpoint_continue), 'rb') as f:
        Trainer = pickle.load(f)
else:
    Trainer = ContrastiveTrainLoop(baseline=args.baseline, device=device, schedule_max_steps=args.epochs)

# load data
if not os.path.exists(args.data_folder) or not os.path.exists(os.path.join(args.data_folder, 'target_curves.npy')) or not os.path.exists(os.path.join(args.data_folder, 'graphs.npy')):
    raise ValueError('All or some of the data does not exist please run Download.py to download the data or provide the correct path.')

data = np.load(os.path.join(args.data_folder, 'target_curves.npy'))
mechanisms = np.load(os.path.join(args.data_folder, 'graphs.npy'), allow_pickle=True)

epochs_remaining = args.epochs - Trainer.current_epoch

for i in range(epochs_remaining):
    hist = Trainer.train(data,mechanisms,args.batch_size,1)
    if (i+1) % args.checkpoint_interval == 0:
        with open(os.path.join(args.checkpoint_folder, args.checkpoint_name + '_epoch' + str(i+1)), 'wb') as f:
            pickle.dump(Trainer, f)
        with open(os.path.join(args.checkpoint_folder, args.checkpoint_name), 'wb') as f:
            pickle.dump(Trainer, f)

with open(os.path.join(args.checkpoint_folder, args.checkpoint_name), 'wb') as f:
    pickle.dump(Trainer, f)