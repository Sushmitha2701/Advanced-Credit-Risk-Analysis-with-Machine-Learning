"""
Comprehensive Model Comparison for Credit Risk Assessment
========================================================

Complete comparison of SVM, Neural Networks, and Random Forest
for credit risk prediction.

Author: Sushmita Singh
MSc Data Analytics, Queen Mary University of London
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    """Comprehensive model comparison framework."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results = {}
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def load_data(self, filepath):
        """Load and prepare dataset."""
        data = pd.read_csv(filepath)
        X = data.drop(columns=['default.payment.next.month'])
        y = data['default.payment.next.month']
        
        print(f"Dataset loaded: {data.shape}")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"Default rate: {y.mean():.1%}")
        
        return X, y
    
    def prepare_data(self, X, y):
        """Prepare data for modeling."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate_traditional_models(self, X_train, X_test, y_train, y_test):
        """Evaluate Random Forest and SVM."""
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=self.random_state)
        }
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
    
    def evaluate_neural_network(self, X_train, X_test, y_train, y_test):
        """Evaluate Neural Network."""
        print("\nTraining Neural Network...")
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(X_train, y_train, epochs=20, batch_size=32, 
                 validation_split=0.2, verbose=0)
        
        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        self.results['Neural Network'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
    
    def plot_comparison(self, save_path=None):
        """Create model comparison visualization."""
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_ylim(0, 1)
            axes[i].set_ylabel(metric.title())
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_results(self):
        """Display comparison results."""
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print("-" * len(model_name))
            for metric, value in metrics.items():
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Find best model for each metric
        print(f"\n{'='*60}")
        print("BEST PERFORMING MODELS BY METRIC")
        print("="*60)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            best_model = max(self.results.keys(), 
                           key=lambda x: self.results[x][metric])
            best_score = self.results[best_model][metric]
            print(f"{metric.replace('_', ' ').title()}: {best_model} ({best_score:.4f})")

def main():
    """Main execution function."""
    print("COMPREHENSIVE CREDIT RISK MODEL COMPARISON")
    print("=" * 50)
    print("MSc Data Analytics Research")
    print("Queen Mary University of London")
    print("=" * 50)
    
    # Initialize comparison framework
    comparison = ModelComparison()
    
    try:
        # Load and prepare data
        X, y = comparison.load_data('data/Credit_default_dataset.csv')
        X_train, X_test, y_train, y_test = comparison.prepare_data(X, y)
        
        # Evaluate all models
        comparison.evaluate_traditional_models(X_train, X_test, y_train, y_test)
        comparison.evaluate_neural_network(X_train, X_test, y_train, y_test)
        
        # Display and save results
        comparison.print_results()
        
        import os
        os.makedirs('results', exist_ok=True)
        comparison.plot_comparison('results/model_comparison.png')
        
        print("\nComparison completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
