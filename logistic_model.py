from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import time

def run_logistic(X_train, X_test, y_train, y_test):
    # Scaling the data for better convergence 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training the model and timing how long it takes
    model = LogisticRegression()
    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train
    
    # Testing the model and timing how long it takes
    start_test = time.time()
    preds = model.predict(X_test_scaled)
    test_time = time.time() - start_test
    
    # Get the metrics of evaluation and time to return
    metrics = {
        'model': 'Logistic Regression',
        'accuracy': accuracy_score(y_test, preds),
        'f1_score': f1_score(y_test, preds),
        'train_time': train_time,
        'test_time': test_time,
        'preds': preds
    }
    return metrics
