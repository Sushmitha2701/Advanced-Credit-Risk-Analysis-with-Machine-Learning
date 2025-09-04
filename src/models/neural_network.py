"""
Neural Network for Credit Risk Assessment
=========================================

Feedforward neural network implementation for credit default prediction.
Part of MSc research comparing AI/ML approaches.

Author: Sushmita Singh
Institution: Queen Mary University of London
Year: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

class CreditRiskNeuralNetwork:
    """Neural Network for credit risk prediction."""
    
    def __init__(self, architecture=[64, 32, 16], learning_rate=0.001, 
                 dropout_rate=0.2, random_state=42):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.performance_metrics = {}
        self.is_trained = False
    
    def build_model(self, input_dim):
        """Build neural network architecture."""
        self.model = Sequential()
        
        # Input layer
        self.model.add(Dense(self.architecture[0], activation='relu', 
                           input_shape=(input_dim,)))
        self.model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.architecture[1:]:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def load_data(self, filepath):
        """Load credit risk dataset."""
        data = pd.read_csv(filepath)
        X = data.drop(columns=['default.payment.next.month'])
        y = data['default.payment.next.month']
        
        print(f"Neural Network - Dataset loaded: {data.shape}")
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Prepare data for neural network training."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train, validation_split=0.2, epochs=50, batch_size=32):
        """Train the neural network."""
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        print("Training Neural Network...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred_prob = self.model.predict(X_test).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        self.performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return self.performance_metrics
    
    def print_results(self):
        """Display results."""
        print("\nNeural Network Performance:")
        print("=" * 35)
        for metric, value in self.performance_metrics.items():
            print(f"{metric.title()}: {value:.4f}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """Main execution function."""
    print("Credit Risk Assessment Using Neural Networks")
    
    nn = CreditRiskNeuralNetwork()
    
    try:
        X, y = nn.load_data('data/Credit_default_dataset.csv')
        X_train, X_test, y_train, y_test = nn.prepare_data(X, y)
        
        nn.train(X_train, y_train, epochs=50)
        nn.evaluate(X_test, y_test)
        nn.print_results()
        
        os.makedirs('results', exist_ok=True)
        nn.plot_training_history('results/nn_training_history.png')
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
