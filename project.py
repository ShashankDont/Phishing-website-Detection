import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data import get_prepared_data
from logistic_model import run_logistic
from random_forest_model import run_random_forest
from knn_model import run_knn

def main():
    # Get the data
    X_train, X_test, y_train, y_test = get_prepared_data()
    
    # Run Logistic Regression
    log_preds = run_logistic(X_train, X_test, y_train, y_test)
    
    # Run Random Forest
    rf_preds = run_random_forest(X_train, X_test, y_train, y_test)
    
    knn_preds = run_knn(X_train, X_test, y_train, y_test)
    
    # Plot the confustion matrices of all models
    print("\nGenerating Confusion Matrices...")
    plot_all_confusion_matrices(y_test, log_preds, rf_preds, knn_preds)


# Plot the confusion matrix for all models
def plot_all_confusion_matrices(y_test, log_preds, rf_preds, knn_preds):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    # Plot Logistic Regression
    cm_log = confusion_matrix(y_test, log_preds)
    disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=['Legitimate', 'Phishing'])
    disp_log.plot(ax=ax1, cmap=plt.cm.Blues)
    ax1.set_title('Logistic Regression')

    # Plot Random Forest
    cm_rf = confusion_matrix(y_test, rf_preds)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Legitimate', 'Phishing'])
    disp_rf.plot(ax=ax2, cmap=plt.cm.Greens)
    ax2.set_title('Random Forest')
    
    # Plot KNN
    cm_knn = confusion_matrix(y_test, knn_preds)
    ConfusionMatrixDisplay(cm_knn, display_labels=['Legit', 'Phish']).plot(ax=ax3, cmap='Purples')
    ax3.set_title('K-Nearest Neighbors')

    plt.tight_layout()
    plt.show()    
    

if __name__ == "__main__":
    main()