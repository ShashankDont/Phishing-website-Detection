from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import time

def run_random_forest(X_train, X_test, y_train, y_test):
    # Initialize the model with 100 trees and as many CPU cores as possible
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    
    # Train the model and timing how long it takes
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    # Testing the model and timing how long it takes
    start_test = time.time()
    preds = model.predict(X_test)
    test_time = time.time() - start_test
    
    # Get the metrics of evaluation and time to return to main
    metrics = {
        'model': 'Random Forest',
        'accuracy': accuracy_score(y_test, preds),
        'f1_score': f1_score(y_test, preds),
        'train_time': train_time,
        'test_time': test_time,
        'preds': preds
    }
    return metrics
