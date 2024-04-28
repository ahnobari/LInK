from LInK.DataUtils import download_data
from LInK.nn import download_checkpoint

import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Download the data and the checkpoint')
parser.add_argument('--data_folder', type=str, default='./Data/', help='The folder to store the data')
parser.add_argument('--test_folder', type=str, default='./TestData/', help='The folder to store the test data')
parser.add_argument('--checkpoint_folder', type=str, default='./Checkpoints/', help='The folder to store the checkpoint')
parser.add_argument('--checkpoint_name', type=str, default='checkpoint.LInK', help='The name of the checkpoint file')
parser.add_argument('--gdrive_id', type=str, default='1vEeAahZ6iivoYLDeHiLN-rrAgA33bq5k', help='The Google Drive ID of the checkpoint file')
args = parser.parse_args()

download_data(args.data_folder, args.test_folder)
download_checkpoint(args.checkpoint_folder, args.checkpoint_name, args.gdrive_id)