# Bachelor Thesis — Evaluating the Impact of Task Adaptive Pre-training (TAPT) in Financial Context: A Comparison of BERT with and without Task Adaptive Pre-training for Named Entity Recognition on the FiNER-139 Dataset
Welcome to the GitHub repository for my bachelor thesis project!  
This repository contains code, notebooks, and documentation related to the development, training, and evaluation of a BERT-based model (with Task-Adaptive Pre-Training and fine-tuning techniques) for evaluating the Impact of Task Adaptive Pre-training (TAPT) in Financial Context.

---

## 📂 Repository Structure

```text
Bachelor_Thesis/
├── 00_FiNER139-TagMerging.ipynb
├── 00a_FiNER139-Text-Extraction.ipynb
├── 00b_FiNER139-Text-Extraction-Analysis.ipynb
├── 04_Model_Comparison_Eval.ipynb
├── Python_01_BERT_TAPT.py
├── Python_02_BERT_base_Finetuning.py
└── Python-03_BERT_TAPT_Finetuning.py
```
---

## 📘 Notebooks

- **00_FiNER139-TagMerging.ipynb**  
  Notebook for merging and preprocessing tag data from Hugging Face datasets: nlpaueb/finer-139.

- **00a_FiNER139-Text-Extraction.ipynb**  
  Workflow for extracting raw textual content from nlpaueb/finer-139.

- **00b_FiNER139-Text-Extraction-Analysis.ipynb**  
  Exploratory data analysis (EDA) of the extracted text, including token distribution, tag frequency, etc.

- **04_Model_Comparison_Eval.ipynb**  
  Comparison and evaluation of various model configurations — metrics, performance, and insights.

---

## 🐍 Python Scripts

- **Python_01_BERT_TAPT.py**  
  Implements Task-Adaptive Pre-Training (TAPT) of BERT on domain-specific data prior to fine-tuning.

- **Python_02_BERT_base_Finetuning.py**  
  Fine-tunes a standard (base) BERT model on the task, following Hugging Face Transformers conventions.

- **Python-03_BERT_TAPT_Finetuning.py**  
  Fine-tunes a TAPT-tuned BERT model on the task, following Hugging Face Transformers conventions.

---

## 🚀 Getting Started

### Requirements

- **Python**: 3.10  
- **PyTorch**: 2.2.0  
- **CUDA**: 12.1.1 (with Ubuntu 22.04 base image: `py3.10-cuda12.1.1-devel-ubuntu22.04`)  

### Required Packages
Install the following dependencies:

```bash
pip install "transformers==4.55.0"
pip install datasets
pip install "accelerate>=0.26.0"
pip install evaluate
pip install seqeval
pip install matplotlib
