import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# TF-IDF, Count Vectorization, N-grams, Word Embeddings
class SentimentFeatureEngineering:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.feature_names = []

    # Load data   
    def load_data(self, file_path):
        print("Loading preprocessed data...")
        df = pd.read_csv(file_path)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sentiment distribution:\n{df['Sentiment'].value_counts()}")
        
        # Handle different column names for tokens
        if 'tokens' not in df.columns:
            if 'tokens_str' in df.columns:
                df['tokens'] = df['tokens_str'].apply(lambda x: x.split() if isinstance(x, str) else [])
                print("✅ Created 'tokens' column from 'tokens_str'")
            else:
                # Create tokens from processed_text if tokens column doesn't exist
                df['tokens'] = df['processed_text'].apply(lambda x: x.split() if isinstance(x, str) else [])
                print("✅ Created 'tokens' column from 'processed_text'")
        
        return df
    
    # Create tf-idf feature function
    def create_tfidf_features(self, texts, max_features=5000, ngram_range=(1, 2)):
        print(f"🔤 Creating TF-IDF features (max_features={max_features}, ngram_range={ngram_range})...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words=None,
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Sample features: {feature_names[:10]}")
        
        return tfidf_matrix, feature_names
    
    # save feature
    def save_features(self, features_dict, filename='features.pkl'):
        print(f"Saving features to {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump(features_dict, f)
        print("Features saved successfully!")
    
    # running tf idf feature method
    def run_feature_engineering(self, file_path, model_path,save_features=True):
        print("Starting Feature Engineering Pipeline...")
        print("=" * 50)
        
        # Load data
        df = self.load_data(file_path)
        
        # Extract texts and tokens
        texts = df['processed_text'].tolist()
        tokenized_texts = df['tokens'].tolist()
        labels = df['Sentiment'].tolist()
        
        # Create all features
        features_dict = {}
        
        # TF-IDF Features
        tfidf_matrix, tfidf_features = self.create_tfidf_features(texts)
        features_dict['tfidf'] = {
            'matrix': tfidf_matrix,
            'feature_names': tfidf_features,
            'vectorizer': self.tfidf_vectorizer
        }
        
        # Labels
        features_dict['labels'] = labels
        features_dict['texts'] = texts
        
        # Save features
        if save_features:
            self.save_features(features_dict, model_path)
        
        print("=" * 50)
        print("Feature Engineering Pipeline Completed!")
        print("\nFeature Summary:")
        print(f"- TF-IDF: {tfidf_matrix.shape[1]} features")
        print(f"- Total samples: {len(texts)}")
        
        return features_dict

# Example usage
if __name__ == "__main__":
    csv_path = "dataset/processed_sentiment_data.csv"
    model_path = "feature_engineering/sentiment_features.pkl"

    # Initialize feature engineering
    fe = SentimentFeatureEngineering()
    
    # Run complete pipeline - adjust filename as needed
    features = fe.run_feature_engineering(csv_path, model_path)
    
    # Train-test split with TF-IDF features
    from sklearn.model_selection import train_test_split
    
    X = features['tfidf']['matrix']
    y = features['labels']
    
    # train:80% 
    # test:20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain-Test Split:")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Training labels distribution: {pd.Series(y_train).value_counts()}")
    print(f"Test labels distribution: {pd.Series(y_test).value_counts()}")