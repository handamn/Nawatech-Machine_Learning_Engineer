import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from preprocessing import IndonesianTextPreprocessor
import sys

class SentimentPredictor:
    def __init__(self, features_path, model_path):
        self.features_path = features_path
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.preprocessor = IndonesianTextPreprocessor()

    def load_model_and_vectorizer(self):
        print("Loading model and vectorizer...")
        # Load model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load features to get vectorizer
        with open(self.features_path, 'rb') as f:
            features = pickle.load(f)
            self.vectorizer = features['tfidf']['vectorizer']
            # Recreate label encoder if available
            if 'labels' in features:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(features['labels'])

        print("Model & vectorizer loaded.")

    #predict sentiment
    def predict(self, texts):
        if not self.model or not self.vectorizer:
            raise RuntimeError("Model or vectorizer not loaded.")

        if isinstance(texts, str):
            texts = [texts]

        print("Applying preprocessing...")
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]

        # Vectorize with TF-IDF
        X = self.vectorizer.transform(preprocessed_texts)

        # Predict
        y_pred = self.model.predict(X)

        # Decode label if needed
        if self.label_encoder:
            try:
                y_pred = self.label_encoder.inverse_transform(y_pred)
            except:
                pass

        # Print result
        for original, cleaned, label in zip(texts, preprocessed_texts, y_pred):
            print("\nInput Tweet:", original)
            print("Preprocessed:", cleaned)
            print("Predicted Sentiment:", label)

        return y_pred



if __name__ == '__main__':
    features_path = "feature_engineering/sentiment_features.pkl"
    model_dir = "model"
    model_name = "xgb_model.pkl"
    model_path = f"{model_dir}/{model_name}"

    predictor = SentimentPredictor(features_path, model_path)
    predictor.load_model_and_vectorizer()

    # if want to use test CLI input: python predict_sentiment.py "Kenapa sinyalnya hilang terus padahal di kota besar?"
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        predictor.predict(input_text)
    else:
        # Test batch
        sample_texts = [
            "Kenapa sinyalnya hilang terus padahal di kota besar?",
            "Terima kasih CS sangat membantu dan ramah.",
            "Saya udah lapor tapi belum ada tindak lanjut."
        ]
        predictor.predict(sample_texts)
