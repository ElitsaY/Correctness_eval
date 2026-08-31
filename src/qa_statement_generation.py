import os

path = ""

CSV_PATH = path+""


import re, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
import sacrebleu
from rouge_score import rouge_scorer as rs

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

"""## 3 · Load & inspect the dataset"""

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

for col in ["question", "answer", "statement"]:
    df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

df = df.dropna(subset=["question", "answer", "statement"])
df = df[(df["question"] != "") & (df["answer"] != "") & (df["statement"] != "")]
df["language"] = df["language"].str.strip()
df = df.reset_index(drop=True)

print(f"Total rows: {len(df)}")
print(df["language"].value_counts().to_string())
df[["language", "question", "answer", "statement"]].head(3)

"""## 4 · Build input / target strings

We format every example as:  
**input** → `question: <Q> answer: <A>`  
**target** → `<statement>`

The colon-prefixed field markers are standard mT5 task prefixes.
"""

df["input_text"]  = "question: " + df["question"] + " answer: " + df["answer"]
df["target_text"] = df["statement"]

def overlap(row):
    src = set(row["input_text"].lower().split())
    tgt = set(row["target_text"].lower().split())
    return len(src & tgt) / len(tgt) if tgt else 0.0

print(f"Mean input→target word-type overlap: {df.apply(overlap, axis=1).mean():.1%}")

"""## 5 · Train / validation / test split"""

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["language"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["language"]
)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

cols = ["input_text", "target_text"]
raw_datasets = DatasetDict({
    split: Dataset.from_pandas(frame[cols].reset_index(drop=True))
    for split, frame in [("train", train_df), ("validation", val_df), ("test", test_df)]
})

"""## 6 · Tokenise"""

MODEL_NAME = "google/mt5-small"
MAX_INPUT  = 512
MAX_TARGET = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    model_inputs = tokenizer(
        batch["input_text"], max_length=MAX_INPUT, truncation=True, padding=False
    )
    labels = tokenizer(
        text_target=batch["target_text"], max_length=MAX_TARGET, truncation=True, padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = raw_datasets.map(
    preprocess, batched=True, remove_columns=["input_text", "target_text"]
)
print(tokenized)



model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
)

rouge = rs.RougeScorer(["rougeL"], use_stemmer=False)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds   = np.where(preds   != -100, preds,   tokenizer.pad_token_id)
    labels  = np.where(labels  != -100, labels,  tokenizer.pad_token_id)
    dec_p   = [s.strip() for s in tokenizer.batch_decode(preds,  skip_special_tokens=True)]
    dec_l   = [s.strip() for s in tokenizer.batch_decode(labels, skip_special_tokens=True)]
    bleu    = sacrebleu.corpus_bleu(dec_p, [dec_l]).score
    rL      = np.mean([rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(dec_l, dec_p)]) * 100
    return {"bleu": round(bleu, 2), "rougeL": round(rL, 2)}

# Create a small subsample of the validation dataset (e.g., 8 examples)
subsample = tokenized["validation"].select(range(min(8, len(tokenized["validation"]))))

# Create a temporary Trainer just for evaluation
temp_trainer = Seq2SeqTrainer(
    model=model,
    args=Seq2SeqTrainingArguments(
        output_dir="./temp_eval",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET,
        generation_num_beams=4,
        per_device_eval_batch_size=8,
        report_to="none"
    ),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Computing metrics on subsample...")
subsample_results = temp_trainer.evaluate(subsample)
print("Subsample Metrics:", subsample_results)

"""## 8 · Training"""

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5-qa-statement",
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    learning_rate=5e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    predict_with_generate=True,
    generation_max_length=MAX_TARGET,
    generation_num_beams=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    fp16=False,
    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    max_grad_norm=1.0,
    logging_steps=5,
    report_to="none",
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    # tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
)

trainer.train()

"""## 9 · Evaluation on the held-out test set"""

test_results = trainer.evaluate(tokenized["test"])
print(test_results)