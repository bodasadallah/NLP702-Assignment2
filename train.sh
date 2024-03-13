#!/bin/bash

#SBATCH --job-name=NLP702-Assignment2 # Job name
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH --qos=gpu-8 
#SBATCH -p gpu                      # Use the gpu partition
##SBATCH --nodelist=ws-l6-017



#########################3 Normal Training ###############################

# python train.py \
# --save_dir='/home/george.ibrahim/Downloads/Semester 2/NLP702/Assignment 2/NLP702-Assignment2' \
# --training_type="finetuning" \
# --epochs=6 \
# --save_steps=500 \
# --eval_steps=500 \
# --logging_steps=500 \
# --report_to="all" \
# --model_name='bert-base-uncased' \
# --per_device_train_batch_size=16 \
# --per_device_val_batch_size=8 \
# --warmup_ratio=0.1 \
# --lr_scheduler_type="linear" \
# --learning_rate=1e-4 \
# echo "ending "




######################### Custom Model Training ############################

# python train.py \
# --training_type="custom" \
# --epochs=6 \
# --save_steps=500 \
# --eval_steps=500 \
# --logging_steps=500 \
# --report_to="all" \
# --model_name='bert-base-uncased' \
# --per_device_train_batch_size=16 \
# --per_device_val_batch_size=8 \
# --warmup_ratio=0.1 \
# --lr_scheduler_type="linear" \
# --hidden_size=768 \
# --num_attention_heads=1 \
# --num_hidden_layers=6 \
# --intermediate_size=3072 \
# --hidden_act="gelu" \ 
# echo "ending "

########################## Distillation  ############################

python train.py \
--save_dir='/home/george.ibrahim/Downloads/Semester 2/NLP702/Assignment 2/NLP702-Assignment2' \
--training_type="distillation" \
--epochs=6 \
--save_steps=500 \
--eval_steps=500 \
--logging_steps=500 \
--report_to="all" \
--model_name='bert-base-uncased' \
--per_device_train_batch_size=16 \
--per_device_val_batch_size=8 \
--warmup_ratio=0.1 \
--lr_scheduler_type="linear" \
--learning_rate=1e-4 \
--stu_hidden_size=768 \
--stu_num_hidden_layers=6 \
--stu_num_attention_heads=1 \
--stu_intermediate_size=3072 \
--stu_hidden_act="gelu" \

# echo "ending "