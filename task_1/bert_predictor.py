import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

class BERTPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.label_map = {0: 'negative', 1: 'positive'}

    def predict(self, text):
        print(f"\nInput: {text}")
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).detach().numpy()[0]
        label_id = np.argmax(probs)
        sentiment = self.label_map[label_id]
        print(f"Prediction: {sentiment} ({probs[label_id]:.4f})")
        return sentiment, probs[label_id]

if __name__ == '__main__':
    model_path = 'model/bert'

    predictor = BERTPredictor(model_path)
    test_text = "Saya udah lapor tapi belum ada tindak lanjut."
    predictor.predict(test_text)
