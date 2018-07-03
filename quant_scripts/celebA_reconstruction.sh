mkdir scripts_celebA

cd src

# Lasso (Wavelet - RGB)
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 20 50 100 200 500 1000 2500 5000 7500 10000 \
    --model-types lasso-wavelet \
    --lmbd 0.00001 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

# Lasso (DCT)
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 20 50 100 200 500 1000 2500 5000 7500 10000 \
    --model-types lasso-dct \
    --lmbd 0.1 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

# DCGAN
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 20 50 100 200 500 1000 2500 5000 7500 10000 \
    --model-types dcgan \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0  \
    --max-update-iter 1500 \
    --num-random-restarts 5 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

# DCGAN l1 wavelet
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 10000 \
    --model-types dcgan_l1_wavelet \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0  \
    --max-update-iter 1500 \
    --lmbd 0.01 \
    --sparse_gen_weight 0.01 \
    --num-random-restarts 5 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 1000 100 \
    --model-types dcgan_l1_wavelet \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0  \
    --max-update-iter 1500 \
    --lmbd 1. \
    --sparse_gen_weight 1. \
    --num-random-restarts 5 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 7500 \
    --model-types dcgan_l1_wavelet \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0  \
    --lmbd 0.01 \
    --sparse_gen_weight 0.01 \
    --max-update-iter 1500 \
    --num-random-restarts 5 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 500 200 \
    --model-types dcgan_l1_wavelet \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0  \
    --max-update-iter 1500 \
    --lmbd 1. \
    --sparse_gen_weight 1. \
    --num-random-restarts 5 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

# DCGAN l1 DCT
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 10000 \
    --model-types dcgan_l1_dct \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0  \
    --max-update-iter 1500 \
    --lmbd 0.01 \
    --sparse_gen_weight 0.01 \
    --num-random-restarts 5 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 1000 100 \
    --model-types dcgan_l1_dct \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0  \
    --max-update-iter 1500 \
    --lmbd 5. \
    --sparse_gen_weight 5. \
    --num-random-restarts 5 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 7500 \
    --model-types dcgan_l1_dct \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0  \
    --lmbd 0.01 \
    --sparse_gen_weight 0.01 \
    --max-update-iter 1500 \
    --num-random-restarts 5 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 500 200 \
    --model-types dcgan_l1_dct \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0  \
    --max-update-iter 1500 \
    --lmbd 5. \
    --sparse_gen_weight 5. \
    --num-random-restarts 5 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 50 \
    --model-types dcgan_l1_dct \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0  \
    --max-update-iter 1500 \
    --lmbd 5. \
    --sparse_gen_weight 5. \
    --num-random-restarts 5 \
    --device_no 0 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA

python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 2500 5000 \
    --model-types dcgan_l1_dct \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0  \
    --lmbd 0.1 \
    --sparse_gen_weight 0.1 \
    --max-update-iter 1500 \
    --num-random-restarts 5 \
    --device_no 1 \
    --experiment_type celebA \
    --scripts-base-dir ../scripts_celebA
