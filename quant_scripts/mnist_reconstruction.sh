mkdir scripts_mnist
mkdir scripts_omni2mnist

cd src

# Lasso 
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 50 100 200 \
    --model-types lasso \
    --lmbd 0.1 \
    --experiment_type mnist \
    --scripts-base-dir ../scripts_mnist

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 300 400 500 750 \
    --model-types lasso \
    --lmbd 0.01 \
    --experiment_type mnist \
    --scripts-base-dir ../scripts_mnist

# VAE

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 50 100 200 300 400 500 750 \
    --model-types vae \
    --zprior_weight 0.1 \
    --max-update-iter 1500 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --ce_weight 0 \
    --lmbd 0 \
    --device_no 0 \
    --experiment_type mnist \
    --scripts-base-dir ../scripts_mnist

# Sparse VAE

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 500 750 \
    --model-types vae_l1 \
    --zprior_weight 0. \
    --max-update-iter 3000 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 0.01 \
    --lmbd 0.01 \
    --device_no 0 \
    --experiment_type mnist \
    --scripts-base-dir ../scripts_mnist

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 200 300 400 \
    --model-types vae_l1 \
    --zprior_weight 0.1 \
    --max-update-iter 1500 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 0.1 \
    --lmbd 0.1 \
    --device_no 0 \
    --experiment_type mnist \
    --scripts-base-dir ../scripts_mnist

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 50 100 \
    --model-types vae_l1 \
    --zprior_weight 0.1 \
    --max-update-iter 1500 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 0.5 \
    --lmbd 0.5 \
    --device_no 0 \
    --experiment_type mnist \
    --scripts-base-dir ../scripts_mnist

# Transfer OMNIGLOT 2 MNIST

python create_scripts.py \
    --input-type transfer-full \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 50 100 200 300 400 500 750 \
    --model-types vae \
    --zprior_weight 0. \
    --max-update-iter 1500 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 0 \
    --lmbd 0 \
    --device_no 0 \
    --experiment_type omni2mnist \
    --scripts-base-dir ../scripts_omni2mnist

python create_scripts.py \
    --input-type transfer-full \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 200 300 400 500 750 \
    --model-types vae_l1 \
    --zprior_weight 0.1 \
    --max-update-iter 1500 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 0.1 \
    --lmbd 0.1 \
    --device_no 0 \
    --experiment_type omni2mnist \
    --scripts-base-dir ../scripts_omni2mnist


python create_scripts.py \
    --input-type transfer-full \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 50 100 \
    --model-types vae_l1 \
    --zprior_weight 0.1 \
    --max-update-iter 1500 \
    --num-random-restarts 10 \
    --learning-rate 0.01 \
    --sparse_gen_weight 0.5 \
    --lmbd 0.5 \
    --device_no 0 \
    --experiment_type omni2mnist \
    --scripts-base-dir ../scripts_omni2mnist
