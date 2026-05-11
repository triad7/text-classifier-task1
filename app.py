from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API is working"}

@app.post("/predict")
def predict(request: TextRequest):
    text_vec = vectorizer.transform([request.text])
    prediction = model.predict(text_vec)[0]
 

    return {
        "input": request.text,
        "predicted_category": prediction
    }

