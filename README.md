# Aspect-Based Sentiment Analysis (ABSA)

## Overview
This project implements an **end-to-end, industry-style Aspect-Based Sentiment Analysis (ABSA) pipeline** using classical NLP techniques, machine learning baselines, and transformer-based models.  
The system extracts **aspect terms** from customer reviews and determines **sentiment polarity at the aspect level**.

The project is structured to mirror **real-world ML systems**, with clear separation between data processing, modeling, inference, and API layers.

---

## Key Features
- SemEval Restaurant Reviews dataset (XML → structured format)
- Robust text cleaning & normalization
- Dependency parsing–based aspect–opinion extraction
- BIO-POL tagging for joint aspect + sentiment modeling
- Classical ML baselines (Logistic Regression, SVM, Random Forest)
- Transformer-based model (BERT BIO-POL)
- Production-ready FastAPI service for inference
- Clean separation of training, inference, and API logic

### Environment Setup

This project supports both pip and conda environments.

- Use `requirements.txt` for pip-based setups
- Use `environment.yml` for conda-based setups



## Project Structure

```t
ABSA/
|-- data/
|   |-- raw/                # Original SemEval XML files
|   |-- processed/          # Cleaned & processed datasets
|
|-- notebooks/
|   |-- 01_data_loading_and_cleanup.ipynb
|   |-- 02_dependency_aspect_extractor.ipynb
|   |-- 03_bio_tagging_and_bert_preparation.ipynb
|   |-- 04_baseline_models.ipynb
|   |-- 05_bert_training.ipynb
|
|-- models/
|   |-- aspect_extraction/        # Trained model artifacts 
|   |-- sentiment_classification/
|
|-- src/
|   |-- inference/
|   |   |-- aspect_extractor.py
|   |
|   |-- config.py
|
|-- api/
|   |-- main.py             # FastAPI service
|   |-- schemas.py
|
|-- requirements.txt
|-- README.md

```





## Modeling Approach

### 1. Baseline Models
To establish reference performance, multiple classical machine learning baselines models were trained:
- Logistic Regression (TF-IDF features)
- Support Vector Machine (linear kernel)
- Random Forest

These baselines provide:
- Fast training and inference
- Interpretability
- A benchmark for comparison against deep learning models



### 2. Transformer Model (BERT BIO-POL)
- Token-level BIO-POL tagging scheme
- Joint learning of:
  - Aspect boundaries
  - Sentiment polarity (POS / NEG / NEU)
- Implemented using HuggingFace Transformers
The training and inference pipeline is fully reproducible in compatible environments such as Linux or cloud-based systems.



## API & Deployment

A FastAPI service exposes the trained model for inference.

### Endpoints
- `GET /health` – Service and model health check
- `POST /predict` – Aspect-wise sentiment prediction

## Model Artifacts

- Trained model weights are not committed to the repository to keep it lightweight.


## Environment Note

- On some Windows systems, PyTorch may fail to load due to binary compatibility issues (e.g., DLL errors).
- The codebase, inference logic, and API design run successfully in modern Linux systems or cloud environments.



### Example input & output
### input
```json
{
  "text": "The food was amazing but the service was slow."
}

### output
{
  "aspects": [
    {"aspect": "food", "sentiment": "positive"},
    {"aspect": "service", "sentiment": "negative"}
  ]
}
```


## Future Enhancements

- Standalone aspect-level sentiment classifier (pipeline comparison)
- Model versioning and experiment tracking
- Dockerized deployment
- Batch inference support


