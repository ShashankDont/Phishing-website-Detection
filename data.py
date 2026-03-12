import pandas as pd
from sklearn.model_selection import train_test_split

def get_prepared_data(filepath='PhishingData.csv'):
    df = pd.read_csv(filepath)
    
    # Change target labels to numeric
    df['Label'] = df['Label'].map({'Phishing': 1, 'Legitimate': 0})
    
    # Remove non-numeric columns to avoid ValueErrors
    X = df.select_dtypes(include=['number']).drop('Label', axis=1)
    y = df['Label']
    
    # Split the data
    return train_test_split(X, y, test_size=0.2)