import pandas as pd
import os

def load_data(judgement_dir, summary_dir):
    data = []
    for filename in os.listdir(judgement_dir):
        if filename.endswith(".txt"):
            judgement_path = os.path.join(judgement_dir, filename)
            summary_path = os.path.join(summary_dir, filename)

            # Only include if both judgement and summary exist
            if os.path.exists(summary_path):
                with open(judgement_path, "r", encoding="utf-8") as j_file, \
                     open(summary_path, "r", encoding="utf-8") as s_file:
                    judgement = j_file.read().strip()
                    summary = s_file.read().strip()
                    data.append({
                        "filename": filename,
                        "text": judgement,
                        "summary": summary
                    })
    return pd.DataFrame(data)

# Paths to your data folders
train_judgements = "C:\\Users\\hp\\Documents\\Programming\\Projects\\Legal Document Summarization\\data\\train_data\\judgement"
train_summaries = "C:\\Users\\hp\\Documents\\Programming\\Projects\\Legal Document Summarization\\data\\train_data\\summary"
test_judgements = "C:\\Users\\hp\\Documents\\Programming\\Projects\\Legal Document Summarization\\data\\test_data\\judgement"
test_summaries = "C:\\Users\\hp\\Documents\\Programming\\Projects\\Legal Document Summarization\\data\\test_data\\summary"

# Load into DataFrames
train_df = load_data(train_judgements, train_summaries)
test_df = load_data(test_judgements, test_summaries)

# Quick check
print("Train samples:", len(train_df))
print("Test samples:", len(test_df))
train_df.head()

# Data cleaning and preprocessing

import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # collapse all whitespace
    text = re.sub(r'\n+', ' ', text)  # remove newlines
    return text.strip()

train_df["cleaned_text"] = train_df["text"].apply(clean_text)
train_df["cleaned_summary"] = train_df["summary"].apply(clean_text)

test_df["cleaned_text"] = test_df["text"].apply(clean_text)
test_df["cleaned_summary"] = test_df["summary"].apply(clean_text)

# Performing EDA

import matplotlib.pyplot as plt

train_df["text_len"] = train_df["cleaned_text"].apply(lambda x: len(x.split()))
train_df["summary_len"] = train_df["cleaned_summary"].apply(lambda x: len(x.split()))

train_df[["text_len", "summary_len"]].hist(bins=30, figsize=(10,4))
plt.show()

print(train_df.isnull().sum())
print(train_df.duplicated().sum())

# Preparing data for Hugging Face Model
from datasets import Dataset
from transformers import BartTokenizer

# Prepare tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Convert to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df[["cleaned_text", "cleaned_summary"]])
test_dataset = Dataset.from_pandas(test_df[["cleaned_text", "cleaned_summary"]])

# Tokenize
def tokenize(batch):
    inputs = tokenizer(batch["cleaned_text"], truncation=True, padding="max_length", max_length=1024)
    targets = tokenizer(batch["cleaned_summary"], truncation=True, padding="max_length", max_length=150)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Fine tuning the summarization model

from transformers import BartForConditionalGeneration, Trainer, TrainingArguments

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
