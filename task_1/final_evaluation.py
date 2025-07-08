import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# COMPARE BETWEEN CLASSIC AND IMPROVEMENT

# Load true labels
df = pd.read_csv("dataset/processed_sentiment_data.csv")
df.dropna(subset=['processed_text', 'Sentiment'], inplace=True)
texts = df['processed_text'].astype(str).tolist()
labels = df['Sentiment'].tolist()

label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(labels)
label_names = label_encoder.classes_

with open("feature_engineering/sentiment_features.pkl", 'rb') as f:
    features = pickle.load(f)
X_tfidf = features['tfidf']['matrix']

def evaluate_and_print(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{model_name}")
    print(classification_report(y_true, y_pred, target_names=label_names))
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    os.makedirs("evaluate", exist_ok=True)
    fname = f"evaluate/cf_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {fname}")

#Classic Model
model_paths = {
    "Naive Bayes": "model/nb_model.pkl",
    "Logistic Regression": "model/logreg_model.pkl",
    "SVM": "model/svm_model.pkl",
    "Random Forest": "model/rf_model.pkl",
    "XGBoost": "model/xgb_model.pkl"
}

for name, path in model_paths.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X_tfidf)
        # special case for XGBoost numeric output
        if isinstance(y_pred[0], (int, np.integer)):
            unique = sorted(set(labels))
            y_pred = [unique[i] for i in y_pred]
        y_pred_enc = label_encoder.transform(y_pred)
        evaluate_and_print(true_labels, y_pred_enc, name)
    else:
        print(f"Model not found: {path}")

#Voting Ensemble
ens_path = "model/voting_model_from_existing.pkl"
if os.path.exists(ens_path):
    with open(ens_path, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X_tfidf)
    y_pred_enc = label_encoder.transform(y_pred)
    evaluate_and_print(true_labels, y_pred_enc, "Voting Ensemble")
else:
    print("Voting ensemble model not found.")

#LSTM
print("\nEvaluating LSTM...")
try:
    from tensorflow.keras.models import load_model
    with open("model/tokenizer_lstm.pkl", 'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model("model/lstm_model.h5")
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=100)
    preds = model.predict(padded, verbose=0)
    y_pred = (preds > 0.5).astype(int).flatten()
    evaluate_and_print(true_labels, y_pred, "LSTM")
except Exception as e:
    print("Failed to evaluate LSTM:", e)

#BERT
print("\nEvaluating BERT...")
try:
    tokenizer = AutoTokenizer.from_pretrained("model/bert")
    model = AutoModelForSequenceClassification.from_pretrained("model/bert")
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, axis=1).numpy()
    evaluate_and_print(true_labels, preds, "BERT")
except Exception as e:
    print("Failed to evaluate BERT:", e)
