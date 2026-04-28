# Phishing Detection: Accuracy vs. Computational Efficiency

This project provides a comparative analysis of four machine learning architectures for detecting phishing websites.
This study focuses on finding a model that is accurate while also being quick in training and testing.


## Models Implemented

| Model | Description |
| :--- | :--- | :--- |
| **Logistic Regression** | Baseline |
| **K-Nearest Neighbors** | Distance-based classification |
| **Random Forest** | Ensemble decision trees |
| **Neural Network** | Deep learning |

## Results Summary
Our findings indicate that while Neural Networks can have high accuracy, they have high training times as well. 
**Random Forest** consistently provides the best balance of predictive accuracy (approx. 97%) and low training and testing times, making it the most suitable candidate for real-time security implementation.

## Getting Started

### Prerequisites
Ensure you have the following installed:
* Python 3.x
* `scikit-learn`
* `pandas`
* `numpy`
* `torch` (for the Neural Network)

### Execution
1. Clone the repository.
2. Ensure the dataset is located in the PhishingData.zip is downloaded.
3. Run the project script:
   ```bash
   python project.py
