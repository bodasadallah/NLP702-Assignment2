from args_parser import get_args
from transformers import BertConfig, AutoModelForSequenceClassification,BertTokenizerFast, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig
# import wandb
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
import torch 
from torch.nn import KLDivLoss, CrossEntropyLoss, Softmax, LogSoftmax


# Define a function to preprocess data for the model
def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(examples["utt"], padding="max_length", truncation=True)
    tokenized['label'] = examples['label']
    return tokenized

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, alpha=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.softmax = Softmax(dim=-1)
        self.log_softmax = LogSoftmax(dim=-1)
        self.kl_div_loss = KLDivLoss(reduction="batchmean")
        self.ce_loss = CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits
        loss_ce = self.ce_loss(student_logits, labels)

        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            loss_kl = self.kl_div_loss(
                self.log_softmax(student_logits / self.temperature),
                self.softmax(teacher_logits / self.temperature)
            )
            loss = (self.alpha * loss_kl) + ((1 - self.alpha) * loss_ce)
        else:
            loss = loss_ce

        return (loss, outputs) if return_outputs else loss



if __name__ == "__main__":
    args = get_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # wandb.init(project=f"nlphw2_{args.model_name}_{args.training_type}" )
    
    print("Loading dataset")
    # Load the Amazon Science Massive dataset (English)
    train_val_test = load_dataset("AmazonScience/massive", 'en-US').rename_columns({"intent":"label"})
    train_dataset = train_val_test["train"]
    val_dataset = train_val_test["validation"]
    test_dataset = train_val_test["test"]
    num_labels=len(train_dataset.features['label'].names)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)


    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # Preprocess training and validation data
    train_dataset = train_dataset.map(tokenize_function,fn_kwargs={"tokenizer":tokenizer}, batched=True)
    val_dataset = val_dataset.map(tokenize_function,fn_kwargs={"tokenizer":tokenizer}, batched=True)
    test_dataset = test_dataset.map(tokenize_function,fn_kwargs={"tokenizer":tokenizer}, batched=True)


    ##################### Custom Bert Model ######################3
    if args.training_type == 'custom':
        config = BertConfig(    hidden_size=args.hidden_size,
                                num_hidden_layers=args.num_hidden_layers,
                                num_attention_heads=args.num_attention_heads,
                                intermediate_size=args.intermediate_size,
                                hidden_act=args.hidden_act,
                            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_config(config).to(device)

        print( '*' * 20,'Training Custom Model', '*' * 20)
        print(model)


    elif args.training_type == 'peft':

        print( '*' * 20, 'Finetuning Using LoRA', '*' * 20)

        # LoRA Config
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_r = args.lora_r
        lora_target_modules = args.lora_target_modules

        lora_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules = lora_target_modules)
        
    
        model = get_peft_model(base_model, lora_config)

        print(model)



    elif args.training_type == 'finetuning':

        print( '*' * 20, 'Normal Finetuning ', '*' * 20)
        model = base_model
        print(model)



    elif args.training_type == 'distillation':
        print("*" * 20, "Distilling Model", "*" * 20)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
        teacher_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher_model.to(device)
 
        student_model_config = BertConfig(
            hidden_size= args.stu_hidden_size, 
            num_hidden_layers= args.stu_num_hidden_layers,
            num_attention_heads= args.stu_num_attention_heads,
            intermediate_size= args.stu_intermediate_size,
            num_labels= num_labels,
            hidden_act=args.stu_hidden_act,
        )
        student_model = AutoModelForSequenceClassification.from_config(student_model_config)
        student_model.to(device)

    save_path = f'{args.save_dir}/{args.model_name}_{args.training_type}'



    training_arguments = TrainingArguments(
        output_dir=save_path,
        logging_dir=args.save_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=args.report_to,
        logging_first_step=True,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    metric = evaluate.load("accuracy")
    
    if args.training_type != "distillation":
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
        )
    elif args.training_type == "distillation":
        trainer = DistillationTrainer(
            model=student_model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            teacher_model=teacher_model, 
            alpha=0.5,  
            temperature=2.0
        )

    # Start training
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model(f"{save_path}/best")  # Adjust save directory
    eval_results = trainer.evaluate(test_dataset)

    print("Evaluation Results:", eval_results)





