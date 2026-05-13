import torch
import torch.nn as nn
import joblib
import time

# Load vectorizer
vectorizer = joblib.load("app/model/vectorizer.pkl")

# Load label encoder
label_encoder = joblib.load("app/model/label_encoder.pkl")

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

# Load model
input_size = len(vectorizer.get_feature_names_out())

num_classes = len(label_encoder.classes_)

model = TextClassifier(input_size, num_classes)

model.load_state_dict(torch.load("app/model/model.pth"))

model.eval()

def predict_category(text: str):

    start = time.time()

    text_vec = vectorizer.transform([text]).toarray()

    text_tensor = torch.tensor(text_vec, dtype=torch.float32)

    with torch.no_grad():

        outputs = model(text_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)

    category = label_encoder.inverse_transform(
        [predicted.item()]
    )[0]

    end = time.time()

    return {
        "category": category,
        "confidence": round(confidence.item(), 4),
        "inference_time_seconds": round(end - start, 5)
    }