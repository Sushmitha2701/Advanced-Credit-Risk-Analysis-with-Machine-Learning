"""
Support Vector Machine for Credit Risk Assessment
================================================

This module implements a Support Vector Machine classifier with RBF kernel
for predicting credit default risk. This implementation is part of MSc research
comparing AI/ML approaches to traditional credit risk assessment methods.

Author: Sushmita Singh
Institution: Queen Mary University of London
Program: MSc Data Analytics
Year: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import joblib
import os
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class CreditRiskSVM:
    """SVM classifier for credit risk prediction with comprehensive evaluation."""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=42):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, 
                        random_state=random_state, probability=True)
        self.scaler = StandardScaler()
        self.performance_metrics = {}
        self.is_trained = False
        
    def load_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and validate credit risk dataset."""
        try:
            data = pd.read_csv(filepath)
            target_col = 'default.payment.next.month'
            
            if target_col not in data.columns:
                raise ValueError(f"Target column '{target_col}' not found")
            
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            print(f"Dataset loaded: {data.shape}")
            print(f"Default rate: {y.mean():.1%}")
            
            return X, y
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split and scale the data."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train):
        """Train the SVM model."""
        print("Training SVM model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed!")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_curve(y_test, y_proba)[2].max()  # Simplified AUC
        }
        
        return self.performance_metrics
    
    def print_results(self):
        """Display evaluation results."""
        if not self.performance_metrics:
            print("No results available. Run evaluate() first.")
            return
        
        print("\nSVM Model Performance:")
        print("=" * 30)
        for metric, value in self.performance_metrics.items():
            print(f"{metric.title()}: {value:.4f}")
    
    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """Plot confusion matrix."""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Default', 'Default'],
                   yticklabels=['Non-Default', 'Default'])
        plt.title('SVM - Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """Main execution function."""
    print("Credit Risk Assessment Using SVM")
    print("=" * 40)
    
    svm = CreditRiskSVM()
    
    # Load and prepare data
    try:
        X, y = svm.load_data('data/Credit_default_dataset.csv')
        X_train, X_test, y_train, y_test = svm.prepare_data(X, y)
        
        # Train and evaluate
        svm.train(X_train, y_train)
        svm.evaluate(X_test, y_test)
        svm.print_results()
        
        # Generate visualizations
        os.makedirs('results', exist_ok=True)
        svm.plot_confusion_matrix(X_test, y_test, 'results/svm_confusion_matrix.png')
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
