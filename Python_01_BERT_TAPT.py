# 01_BERT_TAPT.ipynb
### Purpose: Perform Task-Adaptive Pretraining (TAPT) using MLM on the unlabeled FiNER-139 dataset
from transformers import BertTokenizerFast, BertForMaskedLM, DataCollatorWithPadding, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import torch


### 1. Setup & Imports
# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


### 2. Load Pretrained Model
# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name).to(device)


### 3. Load and Prepare the Unlabeled FiNER-139
# NOTE: Replace the placeholder path with the actual path to your local unlabeled data file
# Assumes one sentence per line in plain text
from datasets import load_from_disk
raw_dataset = load_from_disk("./data/finer-tapt-text-dataset")
print(raw_dataset)


### 4. Tokenization and Preprocessing
# Tokenize with truncation for MLM
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=192)

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=16)
tokenized_dataset = tokenized_dataset.remove_columns("token_type_ids")
print(tokenized_dataset)


### 5. Data Collator for Masked Language Modeling
# Use 15% masking probability (BERT default)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    pad_to_multiple_of=8
)

### 6. Training Arguments
training_args = TrainingArguments(
    overwrite_output_dir=True,
    per_device_train_batch_size=384,
    group_by_length=True,
    num_train_epochs=2,
    dataloader_num_workers=24,
    learning_rate=7.5e-5, 
    weight_decay=0.01,
    lr_scheduler_type="linear",
    save_steps=1000,
    save_total_limit=1,
    output_dir="bert-tapt",
    logging_steps=500,
    fp16=True,
    report_to="none",
    seed=42,        
    data_seed=42,
    optim="adamw_torch_fused"
)

### 7. TAPT Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()


### 8. Save TAPT-Finetuned Model
# Save tokenizer and model to local directory
tapt_model_dir = "bert-tapt"
tokenizer.save_pretrained(tapt_model_dir)
model.save_pretrained(tapt_model_dir)
print(f"TAPT model saved to: {tapt_model_dir}")