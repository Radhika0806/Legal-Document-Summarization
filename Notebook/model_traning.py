# Preparing data for Hugging Face Model
# from datasets import Dataset
from transformers import BartTokenizer

# Prepare tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Convert to HuggingFace Datasets
# train_dataset = Dataset.from_pandas(train_df[["cleaned_text", "cleaned_summary"]])
# test_dataset = Dataset.from_pandas(test_df[["cleaned_text", "cleaned_summary"]])

# Tokenize
def tokenize(batch):
    inputs = tokenizer(batch["cleaned_text"], truncation=True, padding="max_length", max_length=1024)
    targets = tokenizer(batch["cleaned_summary"], truncation=True, padding="max_length", max_length=150)
    inputs["labels"] = targets["input_ids"]
    return inputs

# train_dataset = train_dataset.map(tokenize, batched=True)
# test_dataset = test_dataset.map(tokenize, batched=True)

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
    # train_dataset=train_dataset,
    # eval_dataset=test_dataset
)

trainer.train()
