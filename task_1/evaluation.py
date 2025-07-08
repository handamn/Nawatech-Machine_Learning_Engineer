import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

# Evaluate model created
class ModelEvaluator:
    def __init__(self, features_path, model_path):
        self.features_path = features_path
        self.model_path = model_path
        self.features = None
        self.X = None
        self.y = None

    def load_features(self):
        print("Loading features...")
        with open(self.features_path, 'rb') as f:
            self.features = pickle.load(f)

        self.X = self.features['tfidf']['matrix']
        self.y = self.features['labels']
        print(f"Features loaded. Shape: {self.X.shape}, Labels: {len(self.y)}")

    def evaluate_model(self, model_path, model_name):
        print(f"Evaluating {model_name}...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        y_pred = model.predict(self.X)

        #special case for xgboost due to different approach label use
        if 'xgboost' in model_name.lower() and isinstance(y_pred[0], (int, np.integer)):
            unique_labels_sorted = sorted(set(self.y))
            y_pred = [unique_labels_sorted[i] for i in y_pred]

        acc = accuracy_score(self.y, y_pred)

        print(classification_report(self.y, y_pred))
        print(f"Accuracy: {acc:.4f}")

        self.plot_confusion_matrix(self.y, y_pred, model_name)

        if hasattr(model, "predict_proba"):
            self.plot_roc_curve(model, model_name)

    #create plot confussion matrix
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true)))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        fname = f'evaluate/cf_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved confusion matrix to {fname}")
    
    #create plot ROC curve
    def plot_roc_curve(self, model, model_name):
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(self.y)
        if y_bin.shape[1] == 1:
            y_bin = np.hstack((1 - y_bin, y_bin))

        y_proba = model.predict_proba(self.X)
        fpr, tpr, _ = roc_curve(y_bin[:, 1], y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        fname = f'evaluate/roc_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved ROC curve to {fname}")
    
    #run evaluate model
    def evaluate_all_models(self):
        models = [
            (f"{self.model_path}nb_model.pkl", "Naive Bayes"),
            (f"{self.model_path}logreg_model.pkl", "Logistic Regression"),
            (f"{self.model_path}svm_model.pkl", "SVM"),
            (f"{self.model_path}rf_model.pkl", "Random Forest"),
            (f"{self.model_path}xgb_model.pkl", "XGBoost")
        ]
        for model_path, model_name in models:
            if os.path.exists(model_path):
                self.evaluate_model(model_path, model_name)
            else:
                print(f"Model not found: {model_path}")

if __name__ == '__main__':
    features_path = "feature_engineering/sentiment_features.pkl"
    model_path = "model/"
    evaluator = ModelEvaluator(features_path, model_path)
    evaluator.load_features()
    evaluator.evaluate_all_models()