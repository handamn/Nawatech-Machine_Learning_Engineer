import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class LSTMTrainer:
    def __init__(self, csv_path, model_path, tokenizer_path):
        self.csv_path = csv_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.df = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.max_len = 100
        self.vocab_size = 10000
        self.model = None

    def load_data(self):
        print("Loading dataset...")
        self.df = pd.read_csv(self.csv_path)
        self.df.dropna(subset=['processed_text', 'Sentiment'], inplace=True)
        texts = self.df['processed_text'].astype(str).tolist()
        labels = self.label_encoder.fit_transform(self.df['Sentiment'])
        return texts, labels

    def preprocess(self, texts):
        print("Tokenizing and padding sequences...")
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        return padded

    # Build Model LSTM
    def build_model(self):
        print("Building LSTM model...")
        model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=128, input_length=self.max_len),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return model
    
    # Train LSTM
    def train(self):
        texts, labels = self.load_data()
        X = self.preprocess(texts)
        y = np.array(labels)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        self.build_model()
        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        print("Training...")
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1000,
            batch_size=32,
            callbacks=[es],
            verbose=2
        )

        print(f"Saving model to {self.model_path}...")
        self.model.save(self.model_path)

        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"💾 Tokenizer saved to {self.tokenizer_path}")

if __name__ == '__main__':
    csv_path = "dataset/processed_sentiment_data.csv"
    model_path = "model/lstm_model.h5"
    tokenizer_path='model/tokenizer_lstm.pkl'
    trainer = LSTMTrainer(csv_path, model_path, tokenizer_path)
    trainer.train()

 