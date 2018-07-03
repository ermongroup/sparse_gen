mkdir scripts_omni
mkdir scripts_mnist2omni
cd src

# LASSO

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 50 100 200 \
    --model-types lasso \
    --lmbd 0.1 \
    --experiment_type omni \
    --scripts-base-dir ../scripts_omni


python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 300 400 500 750 \
    --model-types lasso \
    --lmbd 0.01 \
    --experiment_type omni \
    --scripts-base-dir ../scripts_omni

# OMNIGLOT vae

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 50 100 200 300 400 500 750 \
    --model-types vae \
    --zprior_weight 0.1 \
    --max-update-iter 3000 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 0 \
    --lmbd 0 \
    --device_no 0 \
    --experiment_type omni \
    --scripts-base-dir ../scripts_omni

# OMNIGLOT sparse-vae

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 50 \
    --model-types vae_l1 \
    --zprior_weight 0.1 \
    --max-update-iter 3000 \
    --num-random-restarts 10 \
    --learning-rate 0.1 \
    --sparse_gen_weight 1. \
    --lmbd 1. \
    --device_no 0 \
    --experiment_type omni \
    --scripts-base-dir ../scripts_omni

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 100 \
    --model-types vae_l1 \
    --zprior_weight 0.1 \
    --max-update-iter 3000 \
    --num-random-restarts 10 \
    --learning-rate 0.1 \
    --sparse_gen_weight 0.5 \
    --lmbd 0.5 \
    --device_no 0 \
    --experiment_type omni \
    --scripts-base-dir ../scripts_omni

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 200 \
    --model-types vae_l1 \
    --zprior_weight 0.1 \
    --max-update-iter 3000 \
    --num-random-restarts 10 \
    --learning-rate 0.1 \
    --sparse_gen_weight 0.1 \
    --lmbd 0.1 \
    --device_no 0 \
    --experiment_type omni \
    --scripts-base-dir ../scripts_omni

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 300 400 500 \
    --model-types vae_l1 \
    --zprior_weight 0. \
    --max-update-iter 3000 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 0.01 \
    --lmbd 0.01 \
    --device_no 0 \
    --experiment_type omni \
    --scripts-base-dir ../scripts_omni

# TRANSFER MNIST 2 OMNIGLOT

python create_scripts.py \
    --input-type transfer-full \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 300 400 500 \
    --model-types vae_l1 \
    --zprior_weight 0. \
    --max-update-iter 3000 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 0.01 \
    --lmbd 0.01 \
    --device_no 0 \
    --experiment_type mnist2omni \
    --scripts-base-dir ../scripts_mnist2omni

python create_scripts.py \
    --input-type transfer-full \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 100 200 \
    --model-types vae_l1 \
    --zprior_weight 0. \
    --max-update-iter 3000 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 0.1 \
    --lmbd 0.1 \
    --device_no 0 \
    --experiment_type mnist2omni \
    --scripts-base-dir ../scripts_mnist2omni

python create_scripts.py \
    --input-type transfer-full \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 50 \
    --model-types vae_l1 \
    --zprior_weight 0. \
    --max-update-iter 3000 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 1. \
    --lmbd 1. \
    --device_no 0 \
    --experiment_type mnist2omni \
    --scripts-base-dir ../scripts_mnist2omni

python create_scripts.py \
    --input-type transfer-full \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 25 50 100 200 300 400 500 750 \
    --model-types vae \
    --zprior_weight 0. \
    --max-update-iter 3000 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 0. \
    --lmbd 0. \
    --device_no 0 \
    --experiment_type mnist2omni \
    --scripts-base-dir ../scripts_mnist2omni