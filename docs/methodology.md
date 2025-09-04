# Research Methodology

## Dataset Overview
- **Source:** UCI Machine Learning Repository  
- **Records:** 30,000 credit card clients
- **Features:** 25 attributes including demographics and payment history
- **Target:** Binary default classification (20% default rate)

## Model Implementations

### SVM with RBF Kernel
- **Performance:** 81.85% accuracy
- **Hyperparameters:** C=1.0, gamma='scale'

### Neural Network Architecture  
- **Layers:** 64→32→16→1 neurons
- **Training:** 20 epochs, batch size 32
- **Performance:** 81.38% accuracy

### Random Forest (Baseline)
- **Estimators:** 100 decision trees
- **Performance:** 80.72% accuracy

## Key Findings
SVM achieved the highest accuracy, demonstrating superior capability in handling non-linear financial data patterns.
