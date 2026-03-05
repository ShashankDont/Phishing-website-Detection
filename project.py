import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Load the data
df = pd.read_csv('PhishingData.csv')

# 2. Clean the data
df['Label'] = df['Label'].map({'Phishing': 1, 'Legitimate': 0})
X = df.select_dtypes(include=['number']).drop('Label', axis=1)
y = df['Label']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 6. Test and Evaluate the model
predictions = model.predict(X_test_scaled)

print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions):.4f}")
print(f"F1 Score:  {f1_score(y_test, predictions):.4f}")

# Generate the matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Phishing'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Phishing Detection')
plt.show()

