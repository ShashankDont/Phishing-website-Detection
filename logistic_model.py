from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score

def run_logistic(X_train, X_test, y_train, y_test):
    # Scaling the data for better convergence 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training the model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Testing the model and getting the metric for evaluation
    preds = model.predict(X_test_scaled)
    print("--- Logistic Regression Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, preds):.4f}\n")
    
    # Returning the predictions for confusion matrix 
    return preds