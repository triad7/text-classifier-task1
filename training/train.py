import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("training/data.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000)

# Convert text into vectors
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create model
model = LogisticRegression(max_iter=300)

# Train model
model.fit(X_train_vec, y_train)

# Predictions
predictions = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

# Classification report
report = classification_report(y_test, predictions)

# Print metrics
print(report)
print("Accuracy:", accuracy)

# Save metrics
with open("training/metrics.txt", "w") as f:
    f.write(report)
    f.write(f"\nAccuracy: {accuracy}")

# Save model
joblib.dump(model, "app/model/model.pkl")

# Save vectorizer
joblib.dump(vectorizer, "app/model/vectorizer.pkl")

print("Model and vectorizer saved successfully.")