import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class LSTMPredictor:
    def __init__(self, model_path, tokenizer_path, max_len=100):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_len = max_len
        self.model = None
        self.tokenizer = None
        self.label_map = {0: 'negative', 1: 'positive'}
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        print("Loading model and tokenizer...")
        self.model = load_model(self.model_path)
        with open(self.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print("Model and tokenizer loaded.")
    
    #predict using LSTM Model
    def predict(self, text):
        print(f"\nInput: {text}")
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        prob = self.model.predict(padded)[0][0]
        label = 1 if prob >= 0.5 else 0
        sentiment = self.label_map[label]
        print(f"Prediction: {sentiment} ({prob:.4f})")
        return sentiment, prob

if __name__ == '__main__':
    model_path = "model/lstm_model.h5"
    tokenizer_path='model/tokenizer_lstm.pkl'

    predictor = LSTMPredictor(model_path, tokenizer_path)
    test_text = "Saya udah lapor tapi belum ada tindak lanjut."
    predictor.predict(test_text)