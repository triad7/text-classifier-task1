import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("training/data.csv")

# Encode labels
label_encoder = LabelEncoder()

df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label_encoded"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_encoded"]
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000)

X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Convert to tensors
X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_vec, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Neural Network
class TextClassifier(nn.Module):

    def __init__(self, input_size, num_classes):
        super(TextClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x

# Model setup
input_size = X_train_vec.shape[1]

num_classes = len(label_encoder.classes_)

model = TextClassifier(input_size, num_classes)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20

for epoch in range(epochs):

    model.train()

    outputs = model(X_train_tensor)

    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()

with torch.no_grad():

    test_outputs = model(X_test_tensor)

    predictions = torch.argmax(test_outputs, dim=1)

# Metrics
accuracy = accuracy_score(y_test, predictions)

report = classification_report(y_test, predictions)

print(report)

print("Accuracy:", accuracy)

# Save metrics
with open("training/metrics.txt", "w") as f:
    f.write(report)
    f.write(f"\nAccuracy: {accuracy}")

# Save model
torch.save(model.state_dict(), "app/model/model.pth")

# Save vectorizer
joblib.dump(vectorizer, "app/model/vectorizer.pkl")

# Save label encoder
joblib.dump(label_encoder, "app/model/label_encoder.pkl")

print("PyTorch model saved successfully.")