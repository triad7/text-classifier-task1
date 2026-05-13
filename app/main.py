from fastapi import FastAPI, HTTPException

from app.schemas import TextRequest
from app.predictor import predict_category

app = FastAPI(
    title="E-commerce Text Classification API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "status": "API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(request: TextRequest):

    try:
        result = predict_category(request.text)

        return {
            "input": request.text,
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )