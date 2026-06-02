import os
import sys
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Ensure the root directory is in sys.path so we can import 'app'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from app.models import TextClassifier
from app.config import settings

# Dataset class that handles sparse matrix conversion on the fly
class SparseTextDataset(Dataset):
    """
    Custom Dataset to stream TF-IDF representations.
    Keeps vectors sparse in RAM, only converting to dense float tensors
    during batch loading, preventing OOM issues with large datasets.
    """
    def __init__(self, X_sparse, y_labels):
        self.X = X_sparse
        # Convert pandas series or list to tensor
        self.y = torch.tensor(y_labels, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Convert single sparse row to dense on-the-fly
        row_dense = self.X[idx].toarray().squeeze(0)
        return torch.tensor(row_dense, dtype=torch.float32), self.y[idx]

def main():
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "data.csv")
    if not os.path.exists(data_path):
        # Fallback to root data.csv if not in training directory
        data_path = os.path.join(ROOT_DIR, "data.csv")
        
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

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
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_sparse = vectorizer.fit_transform(X_train)
    X_test_sparse = vectorizer.transform(X_test)

    # Create PyTorch datasets and loaders
    train_dataset = SparseTextDataset(X_train_sparse, y_train.values)
    test_dataset = SparseTextDataset(X_test_sparse, y_test.values)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model Setup
    input_size = X_train_sparse.shape[1]
    num_classes = len(label_encoder.classes_)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")

    model = TextClassifier(input_size, num_classes).to(device)

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 20
    print("Starting model training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions = torch.argmax(outputs, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=label_encoder.classes_)

    print("\nTraining complete! Evaluation Metrics:")
    print(report)
    print(f"Accuracy: {accuracy:.4f}")

    # Ensure model folder exists
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = os.path.join(os.path.dirname(__file__), "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {accuracy:.4f}")
    print(f"Saved training metrics to {metrics_file}")

    # Save Model state
    torch.save(model.cpu().state_dict(), settings.MODEL_PATH)
    print(f"Saved PyTorch weights to {settings.MODEL_PATH}")

    # Save vectorizer and label encoder
    joblib.dump(vectorizer, settings.VECTORIZER_PATH)
    print(f"Saved vectorizer to {settings.VECTORIZER_PATH}")

    joblib.dump(label_encoder, settings.LABEL_ENCODER_PATH)
    print(f"Saved label encoder to {settings.LABEL_ENCODER_PATH}")

if __name__ == "__main__":
    main()
