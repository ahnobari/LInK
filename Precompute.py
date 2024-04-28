import pickle
import os
import numpy as np

import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Precompute Embeddings From Checkpoint')
parser.add_argument('--checkpoint_folder', type=str, default='./Checkpoints/', help='The folder to store the checkpoint')
parser.add_argument('--checkpoint_name', type=str, default='checkpoint.LInK', help='The name of the checkpoint file')
parser.add_argument('--output_folder', type=str, default='./Embeddings/', help='The folder to store the embeddings')
parser.add_argument('--output_name', type=str, default='embeddings.npy', help='The name of the embeddings file')
parser.add_argument('--batch_size', type=int, default=10000, help='The batch size for the embeddings')
parser.add_argument('--data_folder', type=str, default='./Data/', help='The folder to store the data')
args = parser.parse_args()

# Load the checkpoint
if not os.path.exists(args.checkpoint_folder) or not os.path.exists(os.path.join(args.checkpoint_folder, args.checkpoint_name)):
    raise ValueError('The checkpoint file does not exist please run Download.py to download the checkpoints or provide the correct path.')

with open(os.path.join(args.checkpoint_folder, args.checkpoint_name), 'rb') as f:
    checkpoint = pickle.load(f)

# Load the data
if not os.path.exists(args.data_folder) or not os.path.exists(os.path.join(args.data_folder, 'target_curves.npy')):
    raise ValueError('The data does not exist please run Download.py to download the data or provide the correct path.')

data = np.load(os.path.join(args.data_folder, 'target_curves.npy'), allow_pickle=True)

# Precompute the embeddings
embs = checkpoint.compute_embeddings_base(data, args.batch_size)

# Save the embeddings
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
    
np.save(os.path.join(args.output_folder, args.output_name), embs)