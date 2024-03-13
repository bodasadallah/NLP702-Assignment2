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


SAVEDIR="/l/users/$USER/nlp702-hw2"

#########################3 Normal Training ###############################

python train.py \
--save_dir=$SAVEDIR \
--training_type="finetuning" \
--epochs=15 \
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
echo "ending "




######################### Custom Model Training ############################

# --model_name='hs=768_nh=32_nl=12' \
# --model_name='hs=1024=128_nl=24' \


# python train.py \
# --save_dir=$SAVEDIR \
# --model_name='hs=768_nh=8_nl=6' \
# --training_type="custom" \
# --epochs=15 \
# --save_steps=500 \
# --eval_steps=500 \
# --logging_steps=500 \
# --report_to="all" \
# --per_device_train_batch_size=32 \
# --per_device_val_batch_size=8 \
# --warmup_ratio=0.1 \
# --lr_scheduler_type="linear" \
# --hidden_act="gelu" \ 
# --intermediate_size=3072 \
# --hidden_size=768 \
# --num_attention_heads=8 \
# --num_hidden_layers=6 \
# echo "ending "

########################## Distillation  ############################



# python train.py \
# --save_dir=/l/users/$USER/nlp702-hw2/ \
# --training_type="custom" \
# --epochs=3 \
# --save_steps=500 \
# --eval_steps=500 \
# --logging_steps=500 \
# --report_to="all" \
# --model_name='bert-base-uncased' \
# --per_device_train_batch_size=8 \
# --per_device_val_batch_size=8 \
# --warmup_ratio=0.1 \
# --lr_scheduler_type="linear" \
# --hidden_size=768 \
# --num_attention_heads=4 \
# --num_hidden_layers=4 \
# --intermediate_size=512 \
# --hidden_act="gelu" \ 
# echo "ending "



########################## Peft ############################

# --save_dir='/l/users/$USER/nlp702-hw2/'

# python train.py \
# --save_dir=/l/users/$USER/nlp702-hw2/ \
# --training_type="peft" \
# --epochs=3 \
# --save_steps=500 \
# --eval_steps=500 \
# --logging_steps=500 \
# --report_to="all" \
# --model_name='bert-base-uncased' \
# --per_device_train_batch_size=8 \
# --per_device_val_batch_size=8 \
# --warmup_ratio=0.1 \
# echo "ending "

########################## Distillation  ############################

# # --model_name='bert-base-uncased' \
# python train.py \
# --save_dir=$SAVEDIR \
# --model_name=$SAVEDIR/bert-base-uncased_finetuning/best \
# --training_type="distillation" \
# --epochs=3 \
# --save_steps=100 \
# --eval_steps=100 \
# --logging_steps=100 \
# --report_to="all" \
# --per_device_train_batch_size=8 \
# --per_device_val_batch_size=8 \
# --warmup_ratio=0.1 \
# --lr_scheduler_type="linear" \
# --learning_rate=1e-4 \
# --stu_hidden_size=768 \
# --stu_num_hidden_layers=4 \
# --stu_num_attention_heads=4 \
# --stu_intermediate_size=512 \
# --stu_hidden_act="gelu" \

# echo "ending "
