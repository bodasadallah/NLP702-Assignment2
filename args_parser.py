import argparse


def get_args():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args, unknown = parser.parse_known_args()
    # args = parser.parse_args()
    return args

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument('--training_type',
                            type=str,
                            default="bert")
    
    parser.add_argument('--save_dir',
                            type=str,
                            default="./experiments")
    parser.add_argument('--dataset_name',
                            type=str,
                            default='old')
    parser.add_argument('--logging_dir',
                            type=str,
                            default="./logs")
    parser.add_argument('--report_to',
                            type=str,
                            default="wandb")
    parser.add_argument('--epochs',
                            type=int,
                            default=3)
    parser.add_argument('--save_steps',
                            type=int,
                            default=400)
    parser.add_argument('--logging_steps',
                            type=int,
                            default=100)
    parser.add_argument('--max_seq_length',
                            type=int,
                            default=256) 
                            
    parser.add_argument('--evaluation_strategy',
                            type=str,
                            default="steps")
    parser.add_argument('--eval_steps',
                            type=int,
                            default=10)
    parser.add_argument('--log_level',
                            type=str,
                            default="info")
    parser.add_argument('--logging_strategy',
                            type=str,
                            default="steps")
    parser.add_argument('--save_total_limit',
                            type=int,
                            default=2)
    parser.add_argument('--model_name',
                            type=str,
                            default='bert-base-uncased')

    parser.add_argument('--output_dir',
                                type=str,
                                default="./experiments")
    
    
    parser.add_argument('--per_device_train_batch_size',
                            type=int,
                            default=4)
    parser.add_argument('--per_device_val_batch_size',
                            type=int,
                            default=2)
    parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=2)
    parser.add_argument('--learning_rate',
                            type=float,
                            default=2e-4)
    parser.add_argument('--max_grad_norm',
                            type=float,
                            default=0.3)

    parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.03)
    parser.add_argument('--lr_scheduler_type',
                            type=str,
                            default="constant")



################################# Custom Model Config ###########################3
    parser.add_argument('--hidden_size',
                            type=int,
                            default=768)
    parser.add_argument('--num_hidden_layers',
                            type=int,
                            default=12)
    parser.add_argument('--num_attention_heads',
                            type=int,
                            default=12)
    parser.add_argument('--intermediate_size',
                            type=int,
                            default=3072)
    parser.add_argument('--hidden_act',
                            type=str,
                            default="gelu")
    
#################################### Distilation Arguments ########################################
    parser.add_argument('--stu_hidden_size', type=int, help="Student model hidden size.")
    parser.add_argument('--stu_num_hidden_layers', type=int, help="Student model number of hidden layers.")
    parser.add_argument('--stu_num_attention_heads', type=int, help="Student model number of attention heads.")
    parser.add_argument('--stu_intermediate_size', type=int, help="Student model intermediate size.")
    parser.add_argument('--stu_hidden_act',
                            type=str,
                            default="gelu")

######################################### Peft Arguments ########################################
        ##################################### LORA #####################################
    parser.add_argument('--lora_alpha',
                            type=int,
                            default=16)
    parser.add_argument('--lora_dropout',
                            type=float,
                            default=0.1)
    parser.add_argument('--lora_r',
                            type=int,
                            default=64)
    parser.add_argument('--bias',
                            type=str,
                            default="none")
    parser.add_argument('--task_type',
                            type=str,
                            default="SEQ_CLS")
    parser.add_argument('--lora_target_modules',
                            type=str,
                            nargs='+',
                            default=[
                                "query",
                                "key",
                                "value",
                                "dense",
                            ])

