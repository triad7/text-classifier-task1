import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import TextRequest
from app.predictor import predict_category, load_model_artifacts

# Setup logging format and log level
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Execute startup routines
    logger.info("Initializing API application...")
    try:
        load_model_artifacts()
        logger.info("Application initialized and ready to serve requests.")
    except Exception as e:
        logger.error(f"Critical error loading model weights at startup: {str(e)}", exc_info=True)
    yield
    # Execute shutdown routines
    logger.info("Shutting down API application...")

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Standard CORS Middleware setup for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in strict production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "status": "API is running",
        "title": settings.APP_TITLE,
        "version": settings.APP_VERSION
    }

@app.get("/health")
def health():
    """
    Comprehensive health check that validates the model components
    are loaded in memory and capable of running inferences.
    """
    from app.predictor import model, vectorizer, label_encoder

    # Check components
    if model is None or vectorizer is None or label_encoder is None:
        logger.error("Health check failed: Model components not loaded.")
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded."
        )

    # Perform a dummy dry run inference to ensure stability
    try:
        predict_category("health check test input")
    except Exception as e:
        logger.error(f"Health check failed during dry-run inference: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Inference validation failed."
        )

    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict")
def predict(request: TextRequest):
    """
    Handles request prediction by routing it to the PyTorch predictor.
    Logs exceptions internally and returns clean error messages.
    """
    try:
        # Log preview of text input
        preview = request.text[:60] + "..." if len(request.text) > 60 else request.text
        logger.info(f"Prediction requested for: '{preview}'")
        
        result = predict_category(request.text)
        
        logger.info(f"Prediction successful. Class: {result['category']} (Confidence: {result['confidence']})")
        return {
            "input": request.text,
            "prediction": result
        }
    except Exception as e:
        # Capture the stack trace inside server logs
        logger.error(f"Error handling prediction request: {str(e)}", exc_info=True)
        # Prevent details leakage to the client
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred while processing the prediction."
        )