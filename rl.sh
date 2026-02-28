#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --config_file ./config/zero2_opt.yaml \
                                    --num_processes 1 --main_process_port 29503 \
                                    rl.py \
                        --model_path /scratch/kl212/minionerec_out/sft_industrial/final_checkpoint \
                        --train_batch_size 4 \
                        --eval_batch_size 8 \
                        --num_train_epochs 2 \
                        --gradient_accumulation_steps 2 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step 0.0999 \
                        --reward_type ranking \
                        --num_generations 4 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir ./output/rl_industrial \
                        --wandb_run_name rl_industrial \
                        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
done
