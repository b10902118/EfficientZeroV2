#!/bin/bash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export IMAGEIO_FFMPEG_EXE=/tmp2/b10902118/micromamba/envs/ezv2/bin/ffmpeg

# # Port for DDP
# export MASTER_PORT='12300'

# Atari
python ez/train.py exp_config=ez/config/exp/atari.yaml 
#python ez/train.py exp_config=ez/config/exp/dmc_state.yaml
