import torch.nn as nn

class TextClassifier(nn.Module):
    """
    Feed-Forward Neural Network for E-commerce text classification.
    Takes a vectorized input (TF-IDF representation) and predicts the category label.
    """
    def __init__(self, input_size: int, num_classes: int):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
