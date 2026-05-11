# Text Classification API using TF-IDF + Logistic Regression

## Project Overview

This project is a machine learning based text classification system built using:

- Python
- Scikit-learn
- FastAPI
- TF-IDF Vectorization
- Logistic Regression

The system classifies e-commerce product descriptions into categories such as:

- Electronics
- Clothing
- Books
- Beauty
- Home

The trained model is exposed as a REST API using FastAPI.

---

# Why I Used TF-IDF Instead of BERT

Initially, transformer-based models like BERT were considered for this project. However, TF-IDF with Logistic Regression was chosen because:

- Faster training time
- Lower system requirements
- Easier to understand and explain
- Simpler deployment
- Better suited for internship-scale prototype development
- Does not require GPU resources

BERT is a powerful deep learning model but requires:

- Large datasets
- Higher memory usage
- Longer training time
- Complex tokenization pipelines
- Transformer architecture knowledge

For this project, the objective was to build a lightweight and functional text classification service quickly and efficiently.

---

# What is TF-IDF?

TF-IDF stands for:

- TF = Term Frequency
- IDF = Inverse Document Frequency

TF-IDF converts text into numerical vectors that machine learning algorithms can understand.

It works by:

1. Counting how often a word appears in a sentence
2. Reducing importance of very common words
3. Increasing importance of unique words

Example:

Input text:

`text
Samsung Galaxy phone with 128GB storage

TF-IDF converts this sentence into a numerical feature vector.

Project Workflow
1. Dataset Creation

A pseudo e-commerce dataset was created containing product descriptions and labels.

Example:

Text	Label
Samsung Galaxy smartphone	Electronics
Cotton printed t-shirt	Clothing
Wooden dining table	Home

The dataset was stored in:

data.csv
2. Data Preprocessing

The dataset was loaded using Pandas.

df = pd.read_csv("data.csv")

The data was split into:

Training data
Testing data

using:

train_test_split()
3. Text Vectorization using TF-IDF

TF-IDF Vectorizer was used to convert text into machine learning features.

vectorizer = TfidfVectorizer(max_features=1000)

The vectorizer learns vocabulary patterns from training text.

4. Model Training

A Logistic Regression classifier was trained on the TF-IDF vectors.

model = LogisticRegression(max_iter=300)

The model learns relationships between words and categories.

Example:

"phone", "laptop" → Electronics
"shirt", "jeans" → Clothing
5. Model Evaluation

Predictions were made on test data.

Evaluation metrics used:

Accuracy
Precision
Recall
F1-score

The model achieved high accuracy on the generated dataset.

6. Saving the Model

The trained model and TF-IDF vectorizer were saved using Joblib.

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

This allows the API to load the trained model without retraining.

FastAPI Integration

FastAPI was used to expose the model as a REST API.

API endpoint:

POST /predict

Example Request:

{
  "text": "Samsung Galaxy phone with 128GB storage"
}

Example Response:

{
  "input": "Samsung Galaxy phone with 128GB storage",
  "predicted_category": "Electronics"
}
Technologies Used
Python
FastAPI
Scikit-learn
Pandas
Joblib
Uvicorn

How to Run the Project =

1. Create Virtual Environment
python -m venv venv
2. Activate Virtual Environment
Git Bash
source venv/Scripts/activate
3. Install Dependencies
pip install fastapi uvicorn scikit-learn pandas joblib
4. Train the Model
python train-bot.py

This generates:

model.pkl
vectorizer.pkl
5. Start FastAPI Server
uvicorn app:app --reload
6. Open Swagger Documentation
http://127.0.0.1:8000/docs
