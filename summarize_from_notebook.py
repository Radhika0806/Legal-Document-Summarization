# Preprocessing the dataset values into dataframe from text files

import os
import pandas as pd
from tqdm import tqdm

def load_data(judgement_dir, summary_dir):
    print("Judgement dir:", judgement_dir)
    print("Summary dir:", summary_dir)

    judgement_files = sorted([f for f in os.listdir(judgement_dir) if f.endswith(".txt")])
    summary_files = set(os.listdir(summary_dir))  # for quick lookup

    print("First 5 Judgement files:", judgement_files[:5])
    print("First 5 Summary files:", list(summary_files)[:5])

    data = []

    for filename in tqdm(judgement_files, desc="Loading data"):
        judgement_path = os.path.join(judgement_dir, filename)
        summary_path = os.path.join(summary_dir, filename)

        with open(judgement_path, "r", encoding="utf-8") as j_file:
            judgement = j_file.read().strip()

        if filename in summary_files:
            with open(summary_path, "r", encoding="utf-8") as s_file:
                summary = s_file.read().strip()

            if judgement and summary:
                data.append({"judgement": judgement, "summary": summary})
        else:
            print(f"[Warning] Summary file missing for: {filename}")

    return pd.DataFrame(data)


# Importing dataset and applying preprocessing

# Paths to data folders
train_judgements = "C:/Users/hp/Documents/Programming/Projects/Legal Document Summarization/data/train-data2/train-data/judgement"
train_summaries = "C:/Users/hp/Documents/Programming/Projects/Legal Document Summarization/data/train-data2/train-data/summary"
test_judgements = "C:/Users/hp/Documents/Programming/Projects/Legal Document Summarization/data/test-data/judgement"
test_summaries = "C:/Users/hp/Documents/Programming/Projects/Legal Document Summarization/data/test-data/summary"

# Load into DataFrames
train_df = load_data(train_judgements, train_summaries)
test_df = load_data(test_judgements, test_summaries)

# Quick check
train_df.head()
print(train_df.columns)


# Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'train_df' is your DataFrame
# 1. Distribution of Judgement Lengths
plt.figure(figsize=(6, 4))
sns.histplot(train_df['judgement'].str.len(), kde=True)
plt.title('Distribution of Judgement Lengths')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.show()


# 2. Distribution of Summary Lengths
plt.figure(figsize=(6,4))
sns.histplot(train_df['summary'].str.len(), kde=True)
plt.title('Distribution of Summary Lengths')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.show()


# 3. Relationship between Judgement and Summary Lengths (Scatter Plot)
plt.figure(figsize=(6,4))
plt.scatter(train_df['judgement'].str.len(), train_df['summary'].str.len(), alpha=0.5)  # alpha for transparency
plt.title('Relationship between Judgement and Summary Lengths')
plt.xlabel('Judgement Length')
plt.ylabel('Summary Length')
plt.show()


#  4. Word Cloud (requires wordcloud library)

from wordcloud import WordCloud

text = " ".join(train_df['judgement'].astype(str))  # Combine all judgements
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud of Judgements')
plt.show()


# prompt: explain what is this wordcloud showing

# The wordcloud visualizes the most frequent words in the "judgement" column of the `train_df` DataFrame.  Larger words indicate higher frequency within the corpus of legal documents.  It provides a quick overview of the dominant themes and terminology present in the dataset.

# prompt: what can we understand from this scatter graph

# The scatter plot titled "Relationship between Judgement and Summary Lengths" visualizes the correlation between the lengths of legal judgements and their corresponding summaries.  Each point on the plot represents a single document pair.
# Here's what we can understand from it:
# * **Positive Correlation:**  The general upward trend of the points suggests a positive correlation between the two lengths.  Longer judgements tend to have longer summaries, and shorter judgements tend to have shorter summaries.
# * **Variability:** While there's an overall positive correlation, there's significant variability.  This means that the relationship isn't perfectly linear.  Some long judgements have relatively short summaries, and vice-versa. This could be due to various factors like the complexity of the case, the amount of detail needed in the summary, or variations in summarizing style.
# * **Outliers:**  Look for points far from the main cluster.  These are outliers, representing cases where the relationship between judgement and summary length is unusual.  They might warrant further investigation to see if there's something unique about those documents.
# In summary, the scatter plot reveals a general tendency for longer summaries to correspond with longer judgements, but with considerable variation, suggesting that other factors influence the summary length beyond just the length of the judgement.

# Converts pandas DataFrames to Hugging Face Dataset objects for compatibility with the transformers library

from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# Initializing a tokenizer from the pre-trained BART model (facebook/bart-large-cnn) to convert text into numerical representations that the model understands.

from transformers import AutoTokenizer

model_name = "facebook/bart-large-cnn"  # Change for second model later
tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenizing the judgement and summary column with a fix length
def preprocess(example):
    inputs = tokenizer(example["judgement"], padding="max_length", truncation=True, max_length=1024)
    targets = tokenizer(example["summary"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_tokenized = train_dataset.map(preprocess, batched=True)
test_tokenized = test_dataset.map(preprocess, batched=True)

# Model Training

from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./results_bart",  # Change for each model
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    # predict_with_generate=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
)

trainer.train()


# prompt: what is happening in the above code?

# The code trains a sequence-to-sequence model (specifically, the BART-large-CNN model from Facebook) for legal document summarization. Let's break down the code step-by-step:

# 1. **Setup and Data Loading:**
#    - Mounts Google Drive to access data files.
#    - Imports necessary libraries (os, pandas, tqdm, datasets, transformers).
#    - The `load_data` function reads pairs of legal judgements and their corresponding summaries from text files located in specified directories.  It handles potential mismatches in filenames and empty files.  The `tqdm` library provides a progress bar during data loading.
#    - It loads training and testing data into pandas DataFrames (`train_df` and `test_df`).
#    - Prints the number of samples loaded for training and testing. Displays the first few rows of the training dataset.
#    - Installs the `datasets` library.
#    - Converts pandas DataFrames to Hugging Face `Dataset` objects for compatibility with the transformers library.

# 2. **Data Preprocessing:**
#    - Initializes a tokenizer from the pre-trained BART model (`facebook/bart-large-cnn`).  Tokenizers convert text into numerical representations that the model understands.
#    - Defines a `preprocess` function that tokenizes both the judgement (input) and the summary (target). It uses padding and truncation to ensure consistent input lengths (1024 for input and 128 for target). Importantly, it assigns the tokenized summary's input IDs to the `labels` field in the input dictionary; this is essential for training seq2seq models using the `Trainer` API.
#    - Applies the `preprocess` function to the train and test datasets using the `map` function of the Hugging Face `Dataset` object, creating tokenized datasets (`train_tokenized` and `test_tokenized`).

# 3. **Model Training:**
#    - Loads the pre-trained BART model.
#    - Sets up `TrainingArguments` to configure the training process: output directory, evaluation strategy (every epoch), saving strategy, learning rate, batch sizes, number of training epochs, weight decay, and logging directory.
#    - Creates a `Trainer` object, passing the model, training arguments, train and evaluation datasets, and the tokenizer.
#    - Calls `trainer.train()` to start the training process.

# **In summary:** The code takes legal judgements and summaries as input data, preprocesses them using a tokenizer, trains a pre-trained sequence-to-sequence model (BART) to learn how to generate summaries from judgements, and saves the trained model and its evaluation results.  The use of Hugging Face's `transformers` and `datasets` libraries streamlines the model training process.


from evaluate import load

rouge = load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds
                                           , skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: v.mid.fmeasure * 100 for k, v in result.items()}

# Evaluate
trainer.evaluate()


# prompt: please explain the ouput

# The code you provided trains a sequence-to-sequence model (BART) for legal document summarization and then evaluates its performance using the ROUGE metric. Let's break down the added evaluation part:

# 1. **Installation of Necessary Libraries:**
#    - `!pip install evaluate`: Installs the `evaluate` library, which provides convenient functions for computing various evaluation metrics.
#    - `!pip install rouge_score`: Installs the `rouge_score` library, specifically for calculating ROUGE scores, a common metric for evaluating text summarization quality.

# 2. **Loading the ROUGE Metric:**
#    - `rouge = load("rouge")`: Loads the ROUGE metric from the `evaluate` library. This metric will be used to compare the generated summaries with the reference summaries.

# 3. **Defining the `compute_metrics` Function:**
#    - This function is crucial for evaluation. It takes the predictions (`preds`) and labels (`labels`) from the model's evaluation output as input.
#    - `decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)`: Decodes the predicted token IDs back into text, removing any special tokens (e.g., padding tokens) that are not part of the actual summary.
#    - `decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)`: Similarly decodes the reference (ground truth) summaries.
#    - `result = rouge.compute(predictions=decoded_preds, references=decoded_labels)`: Computes the ROUGE scores (ROUGE-N, ROUGE-L, and ROUGE-S) by comparing the `decoded_preds` with the `decoded_labels`.  The `rouge.compute` function handles the calculation of the different ROUGE variants.
#    - `return {k: v.mid.fmeasure * 100 for k, v in result.items()}`: Returns a dictionary where the keys are the ROUGE metric names (e.g., "rouge1", "rouge2", "rougeL", "rougeLsum") and the values are the corresponding F1-scores (mid.fmeasure) multiplied by 100.  ROUGE typically reports recall, precision, and F1-measure, and here the F1-measure is being used, which is often considered the most important value.

# 4. **Model Evaluation:**
#    - `trainer.evaluate()`:  This line triggers the evaluation process.  The `Trainer` object internally uses the `compute_metrics` function you defined to calculate and display the ROUGE scores on the test dataset.  The results (ROUGE-1, ROUGE-2, ROUGE-L, etc.) will be printed to the console after execution.  The specific scores will vary based on the model's performance.

# **In summary:** The additions evaluate the trained summarization model. The `compute_metrics` function calculates the ROUGE scores comparing generated summaries to reference summaries and `trainer.evaluate()` triggers the evaluation on the held-out test set, then outputs those ROUGE scores, providing metrics to assess the model’s performance.  The F1 score (fmeasure) is chosen as it's a balanced measure of precision and recall.



# Example of how to show the evaluation metrics as a bar chart
# Assuming 'results' is the dictionary returned by trainer.evaluate()
# results = trainer.evaluate()

# results = {'rouge1': 40.5, 'rouge2': 20.5, 'rougeL': 35.6, 'rougeLsum': 36.0} # Example if trainer.evaluate() does not work

# plt.figure(figsize=(8, 6))
# plt.bar(results.keys(), results.values())
# plt.xlabel("ROUGE Metric")
# plt.ylabel("F1-Score (%)")
# plt.title("ROUGE Evaluation Metrics")
# plt.show()


results = trainer.evaluate()

plt.figure(figsize=(15, 6))
plt.bar(results.keys(), results.values())
plt.xlabel("ROUGE Metric")
plt.ylabel("F1-Score (%)")
plt.title("ROUGE Evaluation Metrics")
plt.show()


import torch
from transformers import pipeline

# Save the model
model_path = "./app/saved_model"
trainer.save_model(model_path)

# Load the saved model for deployment
summarizer = pipeline("summarization", model=model_path, tokenizer=model_name, device=0 if torch.cuda.is_available() else -1)


# Example usage
text_to_summarize = """This Agreement is entered into by and between AlphaTech Solutions, herein referred to as 'Company,' and Beta Innovations, herein referred to as
 'Client,' effective as of the date of the last signature below. The Company agrees to provide software development and consulting services to the Client under the terms
 and conditions outlined herein. The Client shall compensate the Company in accordance with the fee schedule attached as Exhibit A. Both parties agree that all confidential
  information disclosed during the term of this Agreement shall remain confidential and shall not be disclosed to any third party without prior written consent. Either party
   may terminate this Agreement with thirty (30) days’ written notice to the other party. In the event of termination, the Client shall pay the Company for all services
   rendered up to the termination date. This Agreement shall be governed by and construed in accordance with the laws of the State of California, without regard to its conflict
    of law provisions."""

summary = summarizer(text_to_summarize, max_length=130, min_length=30, do_sample=False)

print(summary[0]["summary_text"])

