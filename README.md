 Project Overview
# E-commerce Text Classification API

This project is a production-style Machine Learning API built using FastAPI and PyTorch.

The system classifies e-commerce product descriptions into categories such as:

- Electronics
- Clothing
- Beauty
- Books
- Home

The project uses:

- TF-IDF for feature extraction
- PyTorch Neural Network for classification
- FastAPI for serving predictions as a REST API

The model was trained on a pseudo e-commerce dataset inspired by platforms like Amazon and Flipkart.

# Why TF-IDF Instead of BERT

Although transformer models like BERT provide state-of-the-art NLP performance, this project uses TF-IDF with a PyTorch Neural Network for the following reasons:

- Faster training time
- Lower memory consumption
- Very low inference latency
- Easier deployment
- Simpler architecture for real-time APIs
- Better suited for smaller datasets

This approach provides high accuracy while maintaining lightweight and production-friendly inference performance.

Neural Network Implementation

# Neural Network Architecture

The classification model was implemented using PyTorch.

Architecture:

Input Layer (TF-IDF Features)
        ↓
Linear Layer (1000 → 128)
        ↓
ReLU Activation
        ↓
Output Layer (128 → Number of Classes)

The neural network learns patterns from TF-IDF feature vectors and predicts the most suitable product category.

# Model Evaluation

The system was evaluated on both prediction quality and inference performance.

# Prediction Quality Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-score

Evaluation was performed using Scikit-learn's classification_report().

Example result:

Accuracy: 1.0

The model achieved strong classification performance on the pseudo e-commerce dataset.

# Inference Performance

Inference latency was measured to evaluate real-time API performance.

The API measures:

- Prediction latency
- Confidence score
- Predicted category

Example API response:

{
  "input": "Samsung Galaxy smartphone",
  "prediction": {
    "category": "Electronics",
    "confidence": 0.9821,
    "inference_time_seconds": 0.00143
  }
}

This ensures the system is suitable for low-latency real-time applications.
🚀 Folder Structure
# Project Structure

task1-internship/
│
├── app/
│   ├── main.py
│   ├── predictor.py
│   ├── schemas.py
│   └── model/
│       ├── model.pth
│       ├── vectorizer.pkl
│       └── label_encoder.pkl
│
├── training/
│   ├── train_pytorch.py
│   ├── data.csv
│   └── metrics.txt
│
├── requirements.txt
├── README.md
└── .gitignore
🚀 How to Run the Project
# How to Run

## 1. Create Virtual Environment

bash
python -m venv venv
2. Activate Virtual Environment
Git Bash
source venv/Scripts/activate
3. Install Dependencies
pip install -r requirements.txt
4. Train the Model
python training/train_pytorch.py
5. Start FastAPI Server
uvicorn app.main:app --reload
6. Open Swagger Documentation

http://127.0.0.1:8000/docs


---

# Technologies Used

- Python
- FastAPI
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Uvicorn
 Production Features
# Production-Oriented Features

- Modular project structure
- REST API architecture
- Neural network-based classifier
- TF-IDF feature engineering
- Real-time inference
- Inference latency tracking
- Confidence score prediction
- Input validation using Pydantic
- Health check endpoint
- Model serialization
