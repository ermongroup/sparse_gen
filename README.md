Sparse-Gen
============================================

This repository provides a reference implementation for learning Sparse-Gen models as described in the paper:


> Modeling Sparse Deviations for Compressed Sensing using Generative Models  
Manik Dhar, Aditya Grover, Stefano Ermon  
International Conference on Machine Learning (ICML), 2018  
Paper: https://arxiv.org/abs/1807.01442

## Requirements

The codebase is implemented in Python 2.7. To install the necessary requirements, run the following commands:

```
pip install -r requirements.txt
```

## Setup

The following command will download the CelebA, OMNIGLOT, and MNIST datasets:

```
bash ./setup/download_data.sh
```

The following command will unzip the trained model weights for the experiments:
```
unzip models.zip
```

The following command will create wavelet basis for the celebA experiments

```
python ./src/wavelet_basis.py
```

## Options

Learning and inference of Sparse-Gen models is handled by the `main.py` script which provides the following command line arguments.

```
  --pretrained-model-dir PRETRAINED_MODEL_DIR
                        Directory containing pretrained model
  --dataset DATASET     Dataset to use
  --input-type INPUT_TYPE
                        Where to take input from
  --input-path-pattern INPUT_PATH_PATTERN
                        Pattern to match to get images
  --num-input-images NUM_INPUT_IMAGES
                        number of input images
  --batch-size BATCH_SIZE
                        How many examples are processed together
  --measurement-type MEASUREMENT_TYPE
                        measurement type
  --noise-std NOISE_STD
                        std dev of noise
  --num-measurements NUM_MEASUREMENTS
                        number of gaussian measurements
  --model-types MODEL_TYPES [MODEL_TYPES ...]
                        model(s) used for estimation
  --mloss1_weight MLOSS1_WEIGHT
                        L1 measurement loss weight
  --mloss2_weight MLOSS2_WEIGHT
                        L2 measurement loss weight
  --zprior_weight ZPRIOR_WEIGHT
                        weight on z prior
  --dloss1_weight DLOSS1_WEIGHT
                        -log(D(G(z))
  --dloss2_weight DLOSS2_WEIGHT
                        log(1-D(G(z))
  --sparse_gen_weight SPARSE_GEN_WEIGHT
                        weight for sparse deviations
  --optimizer-type OPTIMIZER_TYPE
                        Optimizer type
  --learning-rate LEARNING_RATE
                        learning rate
  --momentum MOMENTUM   momentum value
  --max-update-iter MAX_UPDATE_ITER
                        maximum updates to z
  --num-random-restarts NUM_RANDOM_RESTARTS
                        number of random restarts
  --decay-lr            whether to decay learning rate
  --lmbd LMBD           lambda : regularization parameter for LASSO
  --lasso-solver LASSO_SOLVER
                        Solver for LASSO
  --const_dummy CONST_DUMMY
                        dummy hack
  --save-images         whether to save estimated images
  --save-stats          whether to save estimated images
  --print-stats         whether to print statistics
  --checkpoint-iter CHECKPOINT_ITER
                        checkpoint every x batches
  --image-matrix IMAGE_MATRIX
                        0 = 00 = no image matrix, 1 = 01 = show image matrix 2
                        = 10 = save image matrix 3 = 11 = save and show image
                        matrix

```

## Examples

### You will need to download the datasets to run the experiments. To run the quantitative experiments as given in the paper, run the scripts in the quant_scripts directory:

```
bash ./quant_scripts/celebA_reconstruction.sh
bash ./quant_scripts/omniglot_reconstruction.sh
bash ./quant_scripts/mnist_reconstruction.sh
```

This will generate the scripts in multiple directories for the required experiments which can be run using the utils/run_sequentially.sh script. The exact commands are as follows:


```
bash ./utils/run_sequentially.sh scripts_mnist
bash ./utils/run_sequentially.sh scripts_mnist2omni
bash ./utils/run_sequentially.sh scritps_omni
bash ./utils/run_sequentially.sh scritps_omni2mnist
bash ./utils/run_sequentially.sh scritps_celebA
```

When all experiments have finished running the graphs can be generated using:

```
bash ./setup/make_graphs.py
```


Portions of the codebase in this repository uses code originally provided in the open-source Compressed Sensing with Generative Model (https://github.com/AshishBora/csgm) repositories. 

## Citing

If you find Sparse-Gen useful in your research, please consider citing the following paper:


>@inproceedings{dhar2018modeling,  
  title={Modeling Sparse Deviations for Compressed Sensing using Generative Models},  
  author={Dhar, Manik and Grover, Aditya and Ermon, Stefano},  
  booktitle={International Conference on Machine Learning},  
  year={2018}}