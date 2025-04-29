#!/bin/sh

export CUDA_VISIBLE_DEVICES=2

folder=better_figure

# for MODEL in xor-1 xor-2 xor-3 xor-ff-1 xor-ff-2 xor-ff-3 xor-ff-4
# for MODEL in xor-ff-1 xor-ff-2 xor-ff-3 xor-ff-4
# do
#   echo ">>> Running $MODEL"
#   python mtkl.py \
#     --k 16 \
#     --N 1000000 \
#     --M_count 20 \
#     --M_eval 1000 \
#     --batch_size 1024 \
#     --exp_folder $folder \
#     --model $MODEL \
#     --ff_a 4 \
#     --ff_b 10
# done

# for MODEL in xor-1 xor-2 xor-3 xor-4 xor-5 xor-6 xor-ff-1 xor-ff-2 xor-ff-3 xor-ff-4
# do
#     for count in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#     do
#     echo ">>> Running $MODEL"
#     python mtkl.py \
#         --k 16 \
#         --N 1000000 \
#         --M_count $count \
#         --M_eval 1000 \
#         --batch_size 2048 \
#         --exp_folder $folder \
#         --model $MODEL \
#         --ff_a 4 \
#         --ff_b 10
#     done
# done
#  8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128
for MODEL in xor-1
do
    for count in 1
    do
        for k in 64
        do
        echo ">>> Running $MODEL"
        python mtkl.py \
            --k $k \
            --N 1000000 \
            --M_count $count \
            --M_eval 1000 \
            --batch_size 2048 \
            --exp_folder $folder \
            --model $MODEL \
            --ff_a 24 \
            --ff_b 40
        done
    done
done