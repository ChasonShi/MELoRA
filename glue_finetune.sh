#!/bin/bash


declare -A epochs=(["mnli"]=30 ["mrpc"]=30 ["qnli"]=25 ["qqp"]=25 ["rte"]=80 ["sst2"]=60 ["stsb"]=40 ["cola"]=8)
declare -A bs=(["mnli"]=128 ["mrpc"]=128 ["qnli"]=128 ["qqp"]=128 ["rte"]=128 ["sst2"]=128 ["stsb"]=128 ["cola"]=64)
declare -A ml=(["mnli"]=256 ["mrpc"]=256 ["qnli"]=256 ["qqp"]=256 ["rte"]=512 ["sst2"]=256 ["stsb"]=256 ["cola"]=256)
declare -A lr=(["mnli"]="5e-4" ["mrpc"]="4e-4" ["qnli"]="4e-4" ["qqp"]="4e-4" ["rte"]="4e-4" ["sst2"]="5e-4" ["stsb"]="4e-4" ["cola"]="4e-4")
declare -A metrics=(["mnli"]="accuracy" ["mrpc"]="accuracy" ["qnli"]="accuracy" ["qqp"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")

export WANDB_MODE=offline
# RTE    2, 490; 277 (5)
# MRPC   3, 668; 408 (8)
# STSB   5, 749; 1, 379 (11)
# CoLA   8, 551; 1, 043 (17)
# QNLI   9, 815; 9, 832 (19)
# SST2  67, 350; 873 (132)
# QQP  363, 870; 40, 431 (711)
# MNLI 392, 702; 9, 815 9, 832 (768)

run(){
  task_name=$1
  learning_rate=${lr[$1]}
  num_train_epochs=${epochs[$1]}
  per_device_train_batch_size=${bs[$1]}
  rank=$2
  l_num=$3
  seed=42
  lora_alpha="16"
  target_modules="query value"
  mode=$4
  lora_dropout=0.05
  lora_bias=none
  lora_task_type=SEQ_CLS
  wandb_project=project_name
  share=false
  wandb_run_name=roberta-lora-${mode}-${task_name}-r-${rank}-n-${l_num}-alpha-16-seed-${seed}-bs-${per_device_train_batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}
  
  exp_dir=../roberta_glue_reproduce/${wandb_run_name}

  CUDA_VISIBLE_DEVICES=0 python ./run_glue_lora.py \
  --model_name_or_path FacebookAI/roberta-base  \
  --task_name ${task_name} \
  --do_train --do_eval \
  --max_seq_length ${ml[$1]} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_train_batch_size} \
  --load_best_model_at_end True --metric_for_best_model ${metrics[$1]} \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${num_train_epochs} \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --weight_decay 0.1 \
  --warmup_ratio 0.06 \
  --logging_steps 10 \
  --seed ${seed} --wandb_project ${wandb_project} \
  --lora_alpha ${lora_alpha} --lora_dropout ${lora_dropout} --lora_bias ${lora_bias} \
  --lora_task_type ${lora_task_type} --target_modules ${target_modules} --rank ${rank} \
  --l_num ${l_num} --mode ${mode} \
  --output_dir ${exp_dir}/model \
  --logging_dir ${exp_dir}/log \
  --run_name ${wandb_run_name} \
  --overwrite_output_dir
}
task_base=('mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'cola')

for task in "${task_base[@]}"; do
    # run $task "8" "1" "base"
    run $task "8" "2" "me"
done