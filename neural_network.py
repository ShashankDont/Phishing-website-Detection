import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import time

# Fully connected Neural Network
# Sigmoid function due to binary classification 
# ReLU to learn non-linear relationships
class PhishingNN(nn.Module):
    def __init__(self, input_size):
        super(PhishingNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

def run_neural_network(X_train, X_test, y_train, y_test, epochs=20, batch_size=32):
    # Using Nvidia GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scaling and prepping the data
    scaler = StandardScaler()
    X_train_scaled = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
    X_test_scaled = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1).to(device)
    
    dataset = TensorDataset(X_train_scaled, y_train_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Intializing the model, loss, and optimizing function
    model = PhishingNN(X_train.shape[1]).to(device)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model and timing the training 
    start_train = time.time()
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_train
    
    # Testing the model and timing the testing
    start_test = time.time()
    model.eval()
    with torch.no_grad():
        raw_preds = model(X_test_scaled)
        nn_preds = (raw_preds > 0.5).float().cpu().numpy().flatten()
    test_time = time.time() - start_test

    # Calculate Final Metrics
    acc = accuracy_score(y_test, nn_preds)
    f1 = f1_score(y_test, nn_preds)

    # Get the metrics of evaluation and time to return to main
    metrics = {
        'model': 'Neural Network',
        'accuracy': acc,
        'f1_score': f1,
        'train_time': train_time,
        'test_time': test_time,
        'preds': nn_preds
    }
    return metrics
