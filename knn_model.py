from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import time

def run_knn(X_train, X_test, y_train, y_test):
    # Scaling the data for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model and time how long it takes
    model = KNeighborsClassifier(n_neighbors=5)
    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train
    
    # Test the model and time how long it takes
    start_test = time.time()
    preds = model.predict(X_test_scaled)
    test_time = time.time() - start_test
    
    
    # Get the metrics of evaluation and time to return to main 
    metrics = {
        'model': 'K-Nearest Neighbors',
        'accuracy': accuracy_score(y_test, preds),
        'f1_score': f1_score(y_test, preds),
        'train_time': train_time,
        'test_time': test_time,
        'preds': preds
    }
    return metrics
