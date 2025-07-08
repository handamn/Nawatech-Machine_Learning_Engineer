import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

"""
Model Training use :
1. Naive Bayes
2. Logistic Regression
3. Support Vector Machine
4. Random Forest
5. XGBoost
"""
class SentimentModelTrainer:
    def __init__(self, features_path, model_path):
        self.features_path = features_path
        self.model_path = model_path
        self.features = None
        self.model = None

    #Load pre-extracted features from pickle file
    def load_features(self):
        print("Loading features...")
        with open(self.features_path, 'rb') as f:
            self.features = pickle.load(f)

        print("Features loaded:")

        for key in self.features.keys():
            val = self.features[key]
            if isinstance(val, dict) and 'matrix' in val:
                print(f"- {key}: shape={val['matrix'].shape}")
            elif hasattr(val, 'shape'):
                print(f"- {key}: shape={val.shape}")
            elif isinstance(val, list):
                print(f"- {key}: length={len(val)}")
            else:
                print(f"- {key}: type={type(val)}")
    
    #Split TF-IDF features into train/test sets function
    def _prepare_data(self):
        X = self.features['tfidf']['matrix']
        y = self.features['labels']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    #Evaluate and print model performance function
    def _evaluate_model(self, y_true, y_pred, model_name):
        print(f"\n Evaluation Report - {model_name}")
        print("-" * 40)
        print(classification_report(y_true, y_pred, digits=4))
        print("Accuracy:", accuracy_score(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true)))

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{self.model_path}confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300)
        plt.show()

    #Run cross-validation and print F1 macro average
    def cross_validate_model(self, model, model_name, cv=5):
        print(f"\n Cross-validating {model_name} with {cv}-fold CV...")
        
        X = self.features['tfidf']['matrix']
        y = self.features['labels']

        if model_name == 'XGBoost':
            le = LabelEncoder()
            y = le.fit_transform(y)

        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
        print(f"F1 Macro Scores: {scores}")
        print(f"Mean F1 Macro: {scores.mean():.4f} ± {scores.std():.4f}")

    #1.Naive Bayes
    def train_naive_bayes(self):
        print("Training Naive Bayes classifier...")
        X_train, X_test, y_train, y_test = self._prepare_data()
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.model = model
        self._evaluate_model(y_test, y_pred, 'Naive Bayes')
        self.cross_validate_model(model, 'Naive Bayes')

    #2.Logistic Regression
    def train_logistic_regression(self):
        print("Training Logistic Regression classifier...")
        X_train, X_test, y_train, y_test = self._prepare_data()
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.model = model
        self._evaluate_model(y_test, y_pred, 'Logistic Regression')
        self.cross_validate_model(model, 'Logistic Regression')

    #3.Support Vector Machine
    def train_svm(self):
        print("Training Support Vector Machine (LinearSVC)...")
        X_train, X_test, y_train, y_test = self._prepare_data()
        model = LinearSVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.model = model
        self._evaluate_model(y_test, y_pred, 'SVM')
        self.cross_validate_model(model, 'SVM')

    #4.Random Forest
    def train_random_forest(self):
        print("Training Random Forest classifier...")
        X_train, X_test, y_train, y_test = self._prepare_data()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.model = model
        self._evaluate_model(y_test, y_pred, 'Random Forest')
        self.cross_validate_model(model, 'Random Forest')

    #5.XGBoost
    def train_xgboost(self):
        if not xgb_available:
            print("XGBoost not installed. Skipping this model.")
            return

        print("Training XGBoost classifier...")
        X_train, X_test, y_train, y_test = self._prepare_data()
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test)
        y_pred_decoded = le.inverse_transform(y_pred)
        y_test_decoded = y_test

        self.model = model
        self._evaluate_model(y_test_decoded, y_pred_decoded, 'XGBoost')
        self.cross_validate_model(model, 'XGBoost')
    
    #Save train moidel
    def save_model(self, filename='sentiment_model.pkl'):
        if self.model:
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"💾 Model saved to {filename}")
        else:
            print("⚠️ No model to save.")

    #train all
    def train_all_models(self):
        self.train_naive_bayes()
        self.save_model(f'{self.model_path}nb_model.pkl')

        self.train_logistic_regression()
        self.save_model(f'{self.model_path}logreg_model.pkl')

        self.train_svm()
        self.save_model(f'{self.model_path}svm_model.pkl')

        self.train_random_forest()
        self.save_model(f'{self.model_path}rf_model.pkl')

        self.train_xgboost()
        self.save_model(f'{self.model_path}xgb_model.pkl')


if __name__ == '__main__':
    features_path = "feature_engineering/sentiment_features.pkl"
    model_path = "model/"
    trainer = SentimentModelTrainer(features_path, model_path)
    trainer.load_features()

    trainer.train_all_models()