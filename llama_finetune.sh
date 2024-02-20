#!/bin/bash
export WANDB_MODE=offline
gpu=0

run(){
  bs=128
  micro_bs=8
  learning_rate='3e-4'
  num_train_epochs=3
  mode=$1
  rank=$2
  l_num=$3
  seed=42
  lora_alpha="16"
  target_name='qv'
  lora_dropout=0.05
  lora_bias=none
  cutoff_len=256
  wandb_project=proejct_name
  wandb_run_name=llama-lora-${target_name}-${mode}-r-${rank}-n-${l_num}-alpha-16-seed-${seed}-bs-${bs}-lr-${learning_rate}-len-${cutoff_len}-epochs-${num_train_epochs}
  echo $wandb_run_name
  exp_dir=./llama-lora/${wandb_run_name}
  mkdir -p $exp_dir
  
  CUDA_VISIBLE_DEVICES=$gpu python llama_finetune.py \
    --base_model= meta-llama/Llama-2-7b-hf \
    --cutoff_len=$cutoff_len \
    --mode=$mode \
    --seed=$seed \
    --group_by_length \
    --lora_r=$rank \
    --lora_n=$l_num \
    --lora_alpha=$lora_alpha \
    --lora_dropout=$lora_dropout \
    --lora_target_modules='[q_proj,v_proj]' \
    --batch_size=$bs \
    --micro_batch_size=$micro_bs \
    --num_epochs=$num_train_epochs \
    --learning_rate=$learning_rate \
    --wandb_project=$wandb_project \
    --wandb_run_name=$wandb_run_name \
    --output_dir=${exp_dir}/model
}

# run LoRA with rank 64
run 'base' 64 1

# run MELoRA with rank 1, num 8
run 'melora' 1 8

# run AdaLoRA with rank 64
run 'adalora' 64 1