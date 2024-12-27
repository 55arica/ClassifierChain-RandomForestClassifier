import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.sparse import hstack


df = pd.read_csv('weatherHistory.csv')

X = df[['Summary', 'Precip Type']]
y = df[['Daily Summary', 'Humidity']]  # Use multiple target columns

encoder = LabelEncoder()

# Encode the 'Precip Type' column
X['Precip Type'] = encoder.fit_transform(X['Precip Type'].astype(str))

# Create a CountVectorizer instance for the 'Summary' column
vectorizer = CountVectorizer(stop_words='english')
summary_features = vectorizer.fit_transform(X['Summary'])

# Combine 'Summary' vectorized features with 'Precip Type'
X_combined = hstack([summary_features, X[['Precip Type']]])

# Encode the target columns
y_encoded = y.copy()
for column in y.columns:
    y_encoded[column] = encoder.fit_transform(y[column].astype(str))


X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42)


base_model = RandomForestClassifier(random_state=42)

# Create a ClassifierChain instance
model = ClassifierChain(base_estimator=base_model, order='random', random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model using classification report
for i, column in enumerate(y.columns):
    print(f"Classification Report for {column}:\n")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i], zero_division=0))
