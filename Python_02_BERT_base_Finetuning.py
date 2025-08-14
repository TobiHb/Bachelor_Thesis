# 02_BERT_base_Finetuning.ipynb
### Purpose: Fine-tune Base model (bert-base) and compare performance against BERT-TAPT


## 1. Setup & Imports
import numpy as np
from datasets import Dataset, DatasetDict
import evaluate
from transformers import BertTokenizerFast, BertForTokenClassification, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


## 2. Load Base Tokenizer & Model
model_name = "bert-base-uncased"  # Change to "bert-base" in Notebook 3
tokenizer = BertTokenizerFast.from_pretrained(model_name)


## 3. Load Labeled FiNER-139 Dataset and Create DatasetDict
train_path = "./data/finer-train.jsonl" 
val_path = "./data/finer-validation.jsonl"
test_path = "./data/finer-test.jsonl"

# Expected format: {"tokens": [...], "ner_tags_str_mapped": [...]}
train_data = Dataset.from_json(train_path)
val_data = Dataset.from_json(val_path)
test_data = Dataset.from_json(test_path)

dataset = DatasetDict({
    "train": train_data,
    "validation": val_data,
    "test": test_data
})
dataset



## 4. Label Mapping & Alignment
all_labels = set()
for split in ["train", "validation", "test"]:
    for ex in dataset[split]["ner_tags_str_mapped"]:
        all_labels.update(ex)
label_list = sorted(all_labels)
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}
num_labels = len(label_list)

# Reinitialize model with correct label size
model = BertForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id_to_label,
    label2id=label_to_id
).to(device)

## 5. Tokenization with Label Alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        max_length=512,
        truncation=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags_str_mapped"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, num_proc=16, remove_columns=dataset["train"].column_names)
tokenized_dataset = tokenized_dataset.remove_columns("token_type_ids")
print(tokenized_dataset["train"])
print(tokenized_dataset["train"][8])



## 6. Data Collator
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
    label_pad_token_id=-100
)

## 7. Metrics: F1 with seqeval
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id_to_label[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [id_to_label[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"]
    }



## 8. Training Configuration
training_args = TrainingArguments(
    per_device_train_batch_size=64,
    num_train_epochs=3,
    learning_rate=3e-5,
    weight_decay=0.01,
    dataloader_num_workers=24,
    dataloader_pin_memory=True,
    group_by_length=True,
    bf16=True,
    logging_steps=1000,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    report_to="none",
    output_dir="bert-base-finetuned",
    seed=42,        
    data_seed=42,
    optim="adamw_torch_fused"
)


## 9. Fine-tuning Base
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

## 10. Final Evaluation
results = trainer.evaluate()
print("📌 Final Validation F1 Score:", results["eval_f1"])

## 11. Save Final Model
model.save_pretrained("bert-base-finetuned")
tokenizer.save_pretrained("bert-base-finetuned")
print("Fine-tuned Base model saved to: bert-base-finetuned")


