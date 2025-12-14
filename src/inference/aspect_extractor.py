import os
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.config import ASPECT_MODEL_DIR


class AspectExtractor:
    """
    Aspect Extraction + Sentiment Inference using BIO-POL BERT model
    """

    def __init__(self, device: str = None):
        self.model_dir = ASPECT_MODEL_DIR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None

        self._load_model()

    def _load_model(self):
        """
        Load model and tokenizer safely.
        """
        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            raise RuntimeError(
                f"Aspect extraction model not found in {self.model_dir}. "
                "Please train the model using Notebook 04."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> List[Dict]:
        """
        Run inference on a single input text.

        Returns:
            [
              {
                "aspect": "food",
                "sentiment": "positive"
              }
            ]
        """
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True
        )

        offsets = encoding.pop("offset_mapping")[0].tolist()
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)[0].tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"][0]
        )

        return self._post_process(text, tokens, predictions, offsets)

    def _post_process(
        self,
        text: str,
        tokens: List[str],
        predictions: List[int],
        offsets: List[List[int]]
    ) -> List[Dict]:
        """
        Convert BIO-POL predictions into structured aspects.
        """
        id2label = self.model.config.id2label

        aspects = []
        current_tokens = []
        current_sentiment = None

        for token, pred_id, (start, end) in zip(tokens, predictions, offsets):
            if start == end:
                continue

            label = id2label[pred_id]

            if label.startswith("B-"):
                if current_tokens:
                    aspects.append({
                        "aspect": "".join(current_tokens),
                        "sentiment": current_sentiment.lower()
                    })
                    current_tokens = []

                current_tokens.append(text[start:end])
                current_sentiment = label.split("-")[1]

            elif label.startswith("I-") and current_tokens:
                current_tokens.append(text[start:end])

            else:
                if current_tokens:
                    aspects.append({
                        "aspect": "".join(current_tokens),
                        "sentiment": current_sentiment.lower()
                    })
                    current_tokens = []
                    current_sentiment = None

        if current_tokens:
            aspects.append({
                "aspect": "".join(current_tokens),
                "sentiment": current_sentiment.lower()
            })

        return aspects
