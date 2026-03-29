import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from tabulate import tabulate 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data import get_prepared_data
from logistic_model import run_logistic
from random_forest_model import run_random_forest
from knn_model import run_knn
from neural_network import run_neural_network

# Set CPU count to 4 for Random Forest
os.environ['LOKY_MAX_CPU_COUNT'] = '4' 

def main():
    # Get the data
    X_train, X_test, y_train, y_test = get_prepared_data()
    
    # Run all models and collect their evaluation metrics
    log_results = run_logistic(X_train, X_test, y_train, y_test)
    rf_results = run_random_forest(X_train, X_test, y_train, y_test)
    knn_results = run_knn(X_train, X_test, y_train, y_test)
    nn_results = run_neural_network(X_train, X_test, y_train, y_test)
    
    # Build and print a table
    all_results = [log_results, rf_results, knn_results, nn_results]
    table_data = []
    for r in all_results:
        table_data.append([
            r['model'], 
            f"{r['accuracy']:.4f}", 
            f"{r['f1_score']:.4f}",
            f"{r['train_time']:.3f}s", 
            f"{r['test_time']:.3f}s"
        ])
    
    # Printing the table
    headers = ["Model", "Accuracy", "F1 Score", "Train Time", "Test Time"]
    print("\n" + "="*65)
    print("MODEL PERFORMANCE & EFFICIENCY COMPARISON")
    print("="*65)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Plot the confusion matrices 
    print("\nGenerating Confusion Matrices...")
    plot_all_confusion_matrices(
        y_test, 
        log_results['preds'], 
        rf_results['preds'], 
        knn_results['preds'], 
        nn_results['preds']
    )

# Plotting Confusion Matrix for all models
def plot_all_confusion_matrices(y_test, log_preds, rf_preds, knn_preds, nn_preds):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Logistic Regression
    cm_log = confusion_matrix(y_test, log_preds)
    ConfusionMatrixDisplay(cm_log, display_labels=['Legit', 'Phishing']).plot(ax=ax1, cmap=plt.cm.Blues)
    ax1.set_title('Logistic Regression')

    # Random Forest
    cm_rf = confusion_matrix(y_test, rf_preds)
    ConfusionMatrixDisplay(cm_rf, display_labels=['Legit', 'Phishing']).plot(ax=ax2, cmap=plt.cm.Greens)
    ax2.set_title('Random Forest')
    
    # KNN 
    cm_knn = confusion_matrix(y_test, knn_preds)
    ConfusionMatrixDisplay(cm_knn, display_labels=['Legit', 'Phishing']).plot(ax=ax3, cmap=plt.cm.Purples)
    ax3.set_title('K-Nearest Neighbors')
    
    # Neural Network
    cm_nn = confusion_matrix(y_test, nn_preds)
    ConfusionMatrixDisplay(cm_nn, display_labels=['Legit', 'Phishing']).plot(ax=ax4, cmap=plt.cm.Oranges)
    ax4.set_title('Neural Network')

    plt.tight_layout()
    plt.show()    

if __name__ == "__main__":
    main()
