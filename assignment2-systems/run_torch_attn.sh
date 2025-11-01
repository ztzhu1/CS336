#!/bin/bash

# for seq_len in 256 1024 4096 8192 16384; do
for seq_len in 256 1024 4096; do
    for d_model in 16 32 64 128; do
        uv run /home/ztzhu/nsight-systems-2025.5.1/bin/nsys profile -o data/profile_time/torch_attn_d_model_${d_model}_seq_len_${seq_len}.nsys-rep python cs336_systems/flash_attention.py --d_model ${d_model} --seq_len ${seq_len} --type torch
    done
done