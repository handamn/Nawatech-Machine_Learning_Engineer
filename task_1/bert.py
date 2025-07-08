import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import os

class BERTTrainer:
    def __init__(self, csv_path, model_name, output_dir):
        self.csv_path = csv_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.df = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.label_encoder = LabelEncoder()
        self.model = None

    def load_data(self):
        print("Loading dataset...")
        self.df = pd.read_csv(self.csv_path)
        self.df.dropna(subset=['processed_text', 'Sentiment'], inplace=True)
        self.df['label'] = self.label_encoder.fit_transform(self.df['Sentiment'])
        return Dataset.from_pandas(self.df[['processed_text', 'label']])

    def tokenize(self, example):
        return self.tokenizer(example['processed_text'], padding='max_length', truncation=True, max_length=128)

    def train(self):
        df = pd.read_csv(self.csv_path)
        df.dropna(subset=['processed_text', 'Sentiment'], inplace=True)
        df['label'] = self.label_encoder.fit_transform(df['Sentiment'])

        # Stratified split via sklearn
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

        # Convert to HuggingFace Dataset
        dataset = {
            'train': Dataset.from_pandas(df_train[['processed_text', 'label']]),
            'test': Dataset.from_pandas(df_test[['processed_text', 'label']])
        }

        # Tokenize
        tokenized = {
            split: ds.map(self.tokenize, batched=True)
            for split, ds in dataset.items()
        }

        print("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_encoder.classes_)
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=4,
            logging_dir=os.path.join(self.output_dir, 'logs'),
        )

        def compute_metrics(pred):
            preds = np.argmax(pred.predictions, axis=1)
            labels = pred.label_ids
            acc = (preds == labels).mean()
            return {"accuracy": acc}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized['train'],
            eval_dataset=tokenized['test'],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        print("Fine-tuning BERT...")
        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model and tokenizer saved to {self.output_dir}")

if __name__ == '__main__':
    csv_path = "dataset/processed_sentiment_data.csv"
    model_name ='indobenchmark/indobert-base-p1'
    model_path = 'model/bert'

    trainer = BERTTrainer(csv_path, model_name, model_path)
    trainer.train()
