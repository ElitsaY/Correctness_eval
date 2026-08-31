import os

model_path = ""

print("Exists:", os.path.exists(model_path))
print("Is directory:", os.path.isdir(model_path))

if os.path.exists(model_path):
    print("Files:")
    print(os.listdir(model_path))
else:
    print("Folder not found.")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = ""

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("Loaded model from:", model_path)
print("Using device:", device)

def generate_statements(inputs, max_new_tokens=128):
    encoded_inputs = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    encoded_inputs = {
        key: value.to(device)
        for key, value in encoded_inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **encoded_inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

import pandas as pd
import os

input_csv_path = ""

df = pd.read_csv(input_csv_path)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())

df.head()

premise_inputs = [
    f"question: {row['question']} answer: {row['gold_answer']}"
    for _, row in df.iterrows()
]

hypothesis_inputs = [
    f"question: {row['question']} answer: {row['answer']}"
    for _, row in df.iterrows()
]

print("Example premise input:")
print(premise_inputs[0])

print("\nExample hypothesis input:")
print(hypothesis_inputs[0])

from tqdm.auto import tqdm

def generate_in_batches(inputs, batch_size=16, max_new_tokens=128):
    all_outputs = []

    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:i + batch_size]

        batch_outputs = generate_statements(
            batch,
            max_new_tokens=max_new_tokens
        )

        all_outputs.extend(batch_outputs)

    return all_outputs

premise_outputs = generate_in_batches(
    premise_inputs,
    batch_size=16,
    max_new_tokens=512
)

hypothesis_outputs = generate_in_batches(
    hypothesis_inputs,
    batch_size=16,
    max_new_tokens=512
)

print("Generated premises:", len(premise_outputs))
print("Generated hypotheses:", len(hypothesis_outputs))

df_output = df.copy()

df_output["premise"] = premise_outputs
df_output["hypothesis"] = hypothesis_outputs

df_output = df_output[
    [
        "question",
        "dataset",
        "gold_answer",
        "answer",
        "hypothesis",
        "premise",
        "label_semantic",
        "label_semantic_manual"
    ]
]

df_output.head()