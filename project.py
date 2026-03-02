import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# 1. Load and Encode
df = pd.read_csv('PhishingData.csv')
df['Label'] = df['Label'].map({'Phishing': 1, 'Legitimate': 0})

# 2. Clean Features
X = df.select_dtypes(include=['number']).drop('Label', axis=1)
y = df['Label']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. FIX: Scale the data (This solves the ConvergenceWarning)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train with more iterations
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# 6. Predict and Calculate Metrics
predictions = model.predict(X_test_scaled)

print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions):.4f}")
print(f"F1 Score:  {f1_score(y_test, predictions):.4f}")

# This gives you a nice summary of everything at once
print("\nFull Classification Report:")
print(classification_report(y_test, predictions))

import numpy as np

# Get the coefficients (weights) assigned to each feature
coefficients = model.coef_[0]
feature_names = X.columns

# Create a dataframe to view them easily
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Weight': coefficients
}).sort_values(by='Weight', ascending=False)

print("\nTop 10 Features Predictors for Phishing:")
print(importance_df.head(10))

print("\nTop 10 Features Predictors for Legitimate:")
print(importance_df.tail(10))