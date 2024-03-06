#!/bin/bash

#SBATCH --job-name=NLP702-Assignment2 # Job name
#SBATCH --error=/home/abdelrahman.sadallah/mbzuai/NLP702-Assignment2/logs/%j%x.err # error file
#SBATCH --output=/home/abdelrahman.sadallah/mbzuai/NLP702-Assignment2/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=46000 # 32 GB of RAM
#SBATCH --nodelist=ws-l6-002





python train.py \
--training_type="finetuning" \
--epochs=3 \
--save_steps=500 \
--eval_steps=500 \
--logging_steps=500 \
--report_to="all" \
--model_name='bert-base-uncased' \
--per_device_train_batch_size=32 \
--per_device_val_batch_size=16 \
--warmup_ratio=0.1 \
--lr_scheduler_type="linear" \
echo "ending "
