# Legal Document Summarization

An NLP pipeline that generates concise summaries of legal court judgements using a fine-tuned BART transformer model, with an interactive Streamlit deployment for real-time inference.

## Overview

Legal documents are often dense and lengthy, making manual summarization time-consuming. This project fine-tunes the `facebook/bart-large-cnn` model on paired legal judgement and summary data to produce abstractive summaries. The trained model is evaluated using ROUGE metrics and deployed through a Streamlit web interface.

## Tech Stack

| Component        | Technology                                    |
|------------------|-----------------------------------------------|
| Language         | Python                                        |
| Base Model       | BART (facebook/bart-large-cnn)                |
| ML Framework     | PyTorch, HuggingFace Transformers             |
| Evaluation       | ROUGE-1, ROUGE-2, ROUGE-L (evaluate library) |
| Data Handling    | HuggingFace Datasets, pandas                  |
| Deployment       | Streamlit                                     |

## Features

- Seq2Seq abstractive summarization fine-tuned specifically for legal court judgements
- Complete training pipeline: data loading, tokenization, fine-tuning, and evaluation
- Quantitative evaluation using ROUGE-1, ROUGE-2, and ROUGE-L metrics
- Exploratory data analysis with distribution plots, scatter plots, and word clouds
- Interactive Streamlit application for real-time document summarization
- Configurable training hyperparameters via HuggingFace Trainer API

## Installation

```bash
git clone <repository-url>
cd "Legal Document Summarization"
pip install -r requirements.txt
```

## Usage

**Training the model:**

```bash
python summarize_from_notebook.py
```

This runs the full pipeline: data loading, tokenization, model fine-tuning (3 epochs), and ROUGE evaluation.

**Running the application:**

```bash
cd app
streamlit run app.py
```

Paste a legal document into the Streamlit interface to generate a summary.

## Project Structure

```
Legal Document Summarization/
|-- summarize_from_notebook.py       # Full training and evaluation pipeline
|-- app/
|   |-- app.py                       # Streamlit deployment application
|   |-- saved_model/                 # Fine-tuned BART model weights
|-- data/
|   |-- train-data2/train-data/      # Training judgement and summary pairs
|   |-- test-data/                   # Test judgement and summary pairs
|-- Notebook/
|   |-- edaAndCleaning.py            # Data loading and exploratory analysis
|   |-- model_traning.py             # Model training script
|-- requirements.txt                 # Python dependencies
```

## Screenshots

| Input Interface | Generated Summary |
|----------------|-------------------|
| ![Input](screenshots/input.png) | ![Summary](screenshots/summary.png) |

> **To add screenshots:** Run `cd app && streamlit run app.py`, paste a legal document, click Summarize, and save screenshots in the `screenshots/` folder.

## Training Configuration

| Parameter        | Value                   |
|------------------|-------------------------|
| Base Model       | facebook/bart-large-cnn |
| Max Input Length  | 1024 tokens             |
| Max Output Length | 128 tokens              |
| Learning Rate    | 2e-5                    |
| Batch Size       | 2                       |
| Epochs           | 3                       |
| Evaluation       | Per-epoch               |

## Evaluation

The fine-tuned model is evaluated against reference summaries using standard ROUGE metrics:

| Metric  | Description                                    |
|---------|------------------------------------------------|
| ROUGE-1 | Unigram overlap between generated and reference |
| ROUGE-2 | Bigram overlap between generated and reference  |
| ROUGE-L | Longest common subsequence similarity           |
