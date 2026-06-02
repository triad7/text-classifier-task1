# E-Commerce Category Classifier API

A high-performance machine learning microservice that classifies e-commerce product listings into standard categories (**Electronics**, **Clothing**, **Home**, **Books**, and **Beauty**). 

The service features a hybrid architecture combining a **PyTorch Feed-Forward Neural Network** with high-efficiency **TF-IDF text vectorization**, fully wrapped inside a **FastAPI web server** and ready for containerized deployment.

---

## 🛠️ Technology Stack

* **Core Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch 2.3+
* **Feature Extraction:** Scikit-learn (TF-IDF vectorizer)
* **Web Framework:** FastAPI (Uvicorn ASGI server)
* **Data Engineering:** Pandas & NumPy
* **Containerization:** Docker & Docker Compose
* **Testing:** Pytest

---

## 💡 Why TF-IDF Instead of BERT

Although transformer models like BERT provide state-of-the-art NLP performance, this project utilizes a high-efficiency TF-IDF feature extractor coupled with a PyTorch Neural Network for the following operational benefits:

- **⚡ Sub-Millisecond Inference Latency:** Runs in under 2 milliseconds, making it ideal for high-throughput, real-time REST APIs.
- **💻 Minimal Memory & Compute Footprint:** The combined weight footprint is under 3MB, allowing seamless execution on cheap, CPU-only cloud environments.
- **🚀 Rapid Training Time:** The entire 20-epoch training cycle completes on a standard CPU in less than 5 seconds.
- **📈 Small Dataset Suitability:** Perfectly handles targeted classification spaces without the risk of heavy overfitting or requiring massive GPU clusters.

---

## 🧠 Neural Network Architecture

The classification model was custom-built using PyTorch:

```
Input Layer (1000 TF-IDF Features)
        ↓
Linear Layer (1000 → 128 hidden units)
        ↓
ReLU Activation
        ↓
Output Layer (128 → 5 Classes)
```

The neural network learns patterns from TF-IDF feature vectors and predicts the most suitable product category with associated confidence scores.

---

## 📂 Repository Structure

```text
├── app/
│   ├── model/                  # Serialized training weights & vectorizers
│   ├── config.py               # Application settings (Pydantic-Settings)
│   ├── main.py                 # FastAPI endpoints & Lifespan management
│   ├── models.py               # PyTorch Neural Network architecture
│   ├── predictor.py            # Prediction pipeline & resource loading
│   └── schemas.py              # Pydantic input/output schemas
├── training/
│   ├── data.csv                # Generated training data
│   ├── train.py                # PyTorch training & evaluation script
│   └── metrics.txt             # Model classification reports
├── tests/
│   └── test_api.py             # Integration & schema validation tests
├── Dockerfile                  # Containerized image recipe
├── docker-compose.yml          # Container service definition
├── generate_data.py            # Synthetic dataset generator
└── requirements.txt            # System dependencies
```

---

## ⚡ Setup & Installation

### 1. Clone & Initialize Environment
Set up a clean virtual environment and install the pinned dependencies:

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Model
The pipeline operates in two clean steps:
```bash
# Step A: Generate synthetic e-commerce dataset (700 rows)
python generate_data.py

# Step B: Train the PyTorch Neural Network
python training/train.py
```
Upon completion, the trained weights (`model.pth`), fitted vectorizer (`vectorizer.pkl`), and encoders are stored directly in `app/model/`.

---

## 🚀 Running the API Server

### Local Development
Launch the Uvicorn server locally with hot-reloading:
```bash
uvicorn app.main:app --reload
```

* **API Docs (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Health Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
* **Base Status:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### Using Docker
Run the entire service inside a secure, lightweight, and non-privileged Docker container:

```bash
# Run via docker-compose
docker-compose up --build
```

---

## 🧪 Testing the API

To verify data validation constraints, home page routing, health status checks, and prediction models:

```bash
pytest
```

---

## 🔌 API Documentation

### **POST** `/predict`
Infers the category of a product given its description.

* **Request Body:**
  ```json
  {
    "text": "Premium Sony Wireless Over-Ear Headphones with Active Noise Cancelling"
  }
  ```

* **Successful Response (`200 OK`):**
  ```json
  {
    "input": "Premium Sony Wireless Over-Ear Headphones with Active Noise Cancelling",
    "prediction": {
      "category": "Electronics",
      "confidence": 0.9984,
      "inference_time_seconds": 0.0014
    }
  }
  ```

* **Validation Error (`422 Unprocessable Entity`):**
  * Inputs must contain at least 3 non-whitespace characters.
  * Inputs are capped at 500 characters to prevent buffer issues.

---

## 🌟 Production-Oriented Features

* **Modular Clean-Architecture Setup** for robust project scaling.
* **Input Schema Constraints** and strict payload validations using Pydantic.
* **Lifespan Context Manager** for fast, asynchronous eager-loading of model components on startup.
* **Low-latency Real-time Inference** tracking internal latency in seconds.
* **Comprehensive Health Probe** with warm dry-run validations on startup.
* **Docker Multi-stage Builds** producing hardened non-root container runner footprints.
