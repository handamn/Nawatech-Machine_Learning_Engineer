import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EnsembleModelTrainer:
    def __init__(self, features_path, model_path):
        self.features_path = features_path
        self.model_path = model_path
        self.X = None
        self.y = None
        self.vectorizer = None
        self.model = None

    def load_features(self):
        print("Loading features and labels...")
        with open(self.features_path, 'rb') as f:
            features = pickle.load(f)
        self.X = features['tfidf']['matrix']
        self.y = features['labels']
        self.vectorizer = features['tfidf']['vectorizer']
        print(f"Features shape: {self.X.shape}")

    def load_model(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _evaluate_model(self, y_true, y_pred, model_name):
        print(f"\nEvaluation Report - {model_name}")
        print(classification_report(y_true, y_pred))
        print("Accuracy:", accuracy_score(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true)))

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{self.model_path}conf_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()

    #use ensemble model from previous model
    def train_voting_from_existing(self):
        print("Loading individual models for Voting Classifier...")
        clf1 = self.load_model(f'{self.model_path}logreg_model.pkl')
        clf2 = self.load_model(f'{self.model_path}svm_model.pkl')
        clf3 = self.load_model(f'{self.model_path}rf_model.pkl')
        clf4 = self.load_model(f'{self.model_path}nb_model.pkl')
        clf5 = self.load_model(f'{self.model_path}xgb_model.pkl')

        voting_clf = VotingClassifier(estimators=[
            ('lr', clf1),
            ('svm', clf2),
            ('rf', clf3),
            ('nb', clf4),
            ('xgb', clf5)
        ], voting='hard')

        print("Fitting Voting Classifier...")
        voting_clf.fit(self.X, self.y)
        y_pred = voting_clf.predict(self.X)

        self.model = voting_clf
        self._evaluate_model(self.y, y_pred, 'Voting Classifier (from existing)')

    def save_model(self, filename='model/voting_model_from_existing.pkl'):
        if self.model:
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {filename}")
        else:
            print("No model to save.")

if __name__ == '__main__':
    features_path = "feature_engineering/sentiment_features.pkl"
    model_path = "model/"
    trainer = EnsembleModelTrainer(features_path, model_path)
    trainer.load_features()
    trainer.train_voting_from_existing()
    trainer.save_model()