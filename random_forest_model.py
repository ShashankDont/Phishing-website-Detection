from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score

def run_random_forest(X_train, X_test, y_train, y_test):
    # Training the model and setting the number of trees to a 100
    # Using all CPU cores for fast training
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Testing the model and showing the metric of evaluation
    preds = model.predict(X_test)
    print("--- Random Forest Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, preds):.4f}")
    
    # Returning the predictions for confusion matrix 
    return preds