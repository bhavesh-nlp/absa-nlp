# Aspect Extraction Model

This directory stores the trained BERT BIO-POL model used for aspect extraction.
Model files are intentionally not committed to the repository to avoid large file sizes and keep the repo lightweight.

# To generate the model locally:
1. Run `notebooks/05_bert_bio_pol_training.ipynb`
2. The trained model will be saved to this directory automatically

The inference API loads the model from this directory at runtime.
