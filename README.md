# LInK: Learning Joint Representations of Design and Performance Spaces through Contrastive Learning for Mechanism Synthesis
LInK is a novel framework that integrates contrastive learning of performance and design space with optimization techniques for solving complex inverse problems in engineering design with discrete and continuous variables. We focus on the path synthesis problem for planar linkage mechanisms in this application. This repo is the official release of the code and data used in this work.

[<img src="https://i.ibb.co/PCWxYQx/ezgif-7-d82a7b1fe9.gif" style="width:100%; max-width:1000px;"/>](https://ahn1376-linkalphabetdemo.hf.space/)

[Simple Fun Alphabet Demo](https://ahn1376-linkalphabetdemo.hf.space/)

## Enviroment and Package Requirements
To Run the code in this demo you will need to replicate the enviromnet we used. This information is available in `environment.yml` file.

## Data & Checkpoint
You can run `Download.py` to automatically download all required data and checkpoint and precomputed embeddings that we use in the paper. Alternatively you can use the arguments in `Download.py` to alter which parts of the data and checkpoint and embeddings you need. Since these files are too large Google Drive may limit the direct download of the files which may lead to `gdown` failing. In this case please use the link below to manually download and add the data to the repo:

[Data & Checkpoint](https://drive.google.com/drive/u/3/folders/1DxIvtFi7Igpn2LlLd-ZgFodC99vuNbrW)

### Download.py
Below is a summary of `Download.py` arguments:

```
Download the data and the checkpoint

options:
  -h, --help            show this help message and exit
  --data_folder DATA_FOLDER
                        The folder to store the data. Default: ./Data/
  --test_folder TEST_FOLDER
                        The folder to store the test data. Default: ./TestData/
  --checkpoint_folder CHECKPOINT_FOLDER
                        The folder to store the checkpoint. Default: ./Checkpoints/
  --checkpoint_name CHECKPOINT_NAME
                        The name of the checkpoint file Default: checkpoint.LInK
  --gdrive_id GDRIVE_ID
                        The Google Drive ID of the checkpoint file
  --download_embedding DOWNLOAD_EMBEDDING
                        Whether to download the embedding file. Default: True
  --embedding_folder EMBEDDING_FOLDER
                        The folder to store the embedding file. Default: ./Embedding/
```

## Hand-Drawing Demo
You can try out the method yourself by running `app.py`. This will launch a gradio demo where you can draw your own target curves and run the model. This will be slower than running the code in pure python without gradio overhead. This requires Gurobi which may not be available based on license availability. However, if you only run the test script later it will not require this.

<img src="https://i.ibb.co/BKS62jG/Drawing.jpg" style="width:100%; max-width:1000px;"/>


### app.py
Below is a summary of `app.py` arguments:

```
usage: app.py [-h] [--port PORT] [--cuda_device CUDA_DEVICE] [--static_folder STATIC_FOLDER] [--checkpoint_folder CHECKPOINT_FOLDER] [--checkpoint_name CHECKPOINT_NAME] [--data_folder DATA_FOLDER] [--embedding_folder EMBEDDING_FOLDER]

options:
  -h, --help            show this help message and exit
  --port PORT           Port number for the local server
  --cuda_device CUDA_DEVICE
                        Cuda devices to use. Default is 0
  --static_folder STATIC_FOLDER
                        Folder to store static files
  --checkpoint_folder CHECKPOINT_FOLDER
                        The folder to store the checkpoint
  --checkpoint_name CHECKPOINT_NAME
                        The name of the checkpoint file
  --data_folder DATA_FOLDER
                        The folder to store the data
  --embedding_folder EMBEDDING_FOLDER
                        The folder to store the embeddings
```

## Test Script
You can run the model without gurobi using the `Test.py` script. This script will take as input a numpy file containing the target curves you want to run the model for and several parameters that are involved in the algorithm.

### Test.py
Below is a summary of `Test.py` arguments:

```
usage: Test.py [-h] [--cuda_device CUDA_DEVICE] [--checkpoint_folder CHECKPOINT_FOLDER] [--checkpoint_name CHECKPOINT_NAME] [--data_folder DATA_FOLDER] [--embedding_folder EMBEDDING_FOLDER] [--test_curves TEST_CURVES] [--save_name SAVE_NAME]
               [--vis_folder VIS_FOLDER] [--vis_candidates VIS_CANDIDATES] [--max_joint MAX_JOINT] [--optim_timesteps OPTIM_TIMESTEPS] [--top_n TOP_N] [--init_optim_iters INIT_OPTIM_ITERS] [--top_n_level2 TOP_N_LEVEL2] [--CD_weight CD_WEIGHT]
               [--OD_weight OD_WEIGHT] [--BFGS_max_iter BFGS_MAX_ITER] [--n_repos N_REPOS] [--BFGS_lineserach_max_iter BFGS_LINESERACH_MAX_ITER] [--BFGS_line_search_mult BFGS_LINE_SEARCH_MULT] [--curve_size CURVE_SIZE] [--smoothing SMOOTHING]
               [--n_freq N_FREQ]

options:
  -h, --help            show this help message and exit
  --cuda_device CUDA_DEVICE
                        Cuda devices to use. Default is 0
  --checkpoint_folder CHECKPOINT_FOLDER
                        The folder to store the checkpoint
  --checkpoint_name CHECKPOINT_NAME
                        The name of the checkpoint file
  --data_folder DATA_FOLDER
                        The folder to store the data
  --embedding_folder EMBEDDING_FOLDER
                        The folder to store the embeddings
  --test_curves TEST_CURVES
                        The path to a npy file containing the test curves you want to run LInK on. Default is ./TestData/alphabet.npy
  --save_name SAVE_NAME
                        The name of the file to save the synthesized mechanisms. Default is results.npy
  --vis_folder VIS_FOLDER
                        The folder to save the visualization of the synthesized mechanisms. Default is ./Vis/ only used if vis_candidates is True.
  --vis_candidates VIS_CANDIDATES
                        Whether to visualize the retrieved candidates. Default is True.
  --max_joint MAX_JOINT
                        The maximum number of joints to use in the synthesis. Default is 20
  --optim_timesteps OPTIM_TIMESTEPS
                        The number of timesteps to use in the solver. Default is 2000
  --top_n TOP_N         The number of paths to keep in the first level of the optimization. Default is 300
  --init_optim_iters INIT_OPTIM_ITERS
                        The number of iterations to run the first level of optimization. Default is 10
  --top_n_level2 TOP_N_LEVEL2
                        The number of paths to keep in the second level of the optimization. Default is 30
  --CD_weight CD_WEIGHT
                        The weight of the chamfer distance in the optimization. Default is 1.0
  --OD_weight OD_WEIGHT
                        The weight of the ordered distance in the optimization. Default is 0.25
  --BFGS_max_iter BFGS_MAX_ITER
                        The maximum number of iterations to run the BFGS optimization. Default is 150
  --n_repos N_REPOS     The number of repositioning to use in the optimization. Default is 1
  --BFGS_lineserach_max_iter BFGS_LINESERACH_MAX_ITER
                        The maximum number of line search iterations to use in the optimization. Default is 10
  --BFGS_line_search_mult BFGS_LINE_SEARCH_MULT
                        The multiplier to use in the line search. Default is 0.5
  --curve_size CURVE_SIZE
                        The number of points to uniformize the curves. Default is 200
  --smoothing SMOOTHING
                        Whether to smooth the curves. Default is True. Recommended to keep it True only for hand drawn curves or curves with high noise
  --n_freq N_FREQ       The number of frequencies to use in the smoothing. Default is 7
  ```
## Training 
To train the model from scratch you can run `Train.py`:

```
usage: Train.py [-h] [--cuda_device CUDA_DEVICE] [--checkpoint_folder CHECKPOINT_FOLDER] [--checkpoint_name CHECKPOINT_NAME] [--checkpoint_interval CHECKPOINT_INTERVAL] [--data_folder DATA_FOLDER] [--baseline BASELINE] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--checkpoint_continue CHECKPOINT_CONTINUE]

options:
  -h, --help            show this help message and exit
  --cuda_device CUDA_DEVICE
                        Cuda devices to use. Default is 0
  --checkpoint_folder CHECKPOINT_FOLDER
                        The folder to store the checkpoint
  --checkpoint_name CHECKPOINT_NAME
                        The name of the checkpoint file to save, this will be used with _epoch for checkpointing and the main file will always be the latest checkpoint
  --checkpoint_interval CHECKPOINT_INTERVAL
                        The epoch interval to save the checkpoint Default is 1
  --data_folder DATA_FOLDER
                        The folder to store the data
  --baseline BASELINE   If set to GCN or GIN the model will train a Naive GCN or GIN model instead of GHOP. Default is GHOP
  --epochs EPOCHS       The number of epochs to train the model. Default is 50
  --batch_size BATCH_SIZE
                        The batch size for training. Default is 256
  --checkpoint_continue CHECKPOINT_CONTINUE
                        The checkpoint file to continue training from. This should be in the checkpoint folder. Default is empty
```

After training you must precompute the embeddings and use the appropriate arguments for checkpoints. To get precomputed embeddings run `Precompute.py`:

```
usage: Precompute.py [-h] [--checkpoint_folder CHECKPOINT_FOLDER] [--checkpoint_name CHECKPOINT_NAME] [--output_folder OUTPUT_FOLDER] [--output_name OUTPUT_NAME] [--batch_size BATCH_SIZE] [--data_folder DATA_FOLDER]

Precompute Embeddings From Checkpoint

options:
  -h, --help            show this help message and exit
  --checkpoint_folder CHECKPOINT_FOLDER
                        The folder to store the checkpoint
  --checkpoint_name CHECKPOINT_NAME
                        The name of the checkpoint file
  --output_folder OUTPUT_FOLDER
                        The folder to store the embeddings
  --output_name OUTPUT_NAME
                        The name of the embeddings file
  --batch_size BATCH_SIZE
                        The batch size for the embeddings
  --data_folder DATA_FOLDER
                        The folder to store the data
```