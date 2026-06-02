import torch
import joblib
import time
import logging
from app.models import TextClassifier
from app.config import settings

logger = logging.getLogger(__name__)

# Global variables for loaded artifacts
vectorizer = None
label_encoder = None
model = None

def load_model_artifacts():
    """
    Safely load model artifacts from paths defined in settings.
    Forces weights onto the CPU device to remain compatible with CPU containers.
    """
    global vectorizer, label_encoder, model
    
    # Avoid reloading if already loaded
    if model is not None:
        return

    try:
        logger.info(f"Loading vectorizer from: {settings.VECTORIZER_PATH}")
        vectorizer = joblib.load(settings.VECTORIZER_PATH)

        logger.info(f"Loading label encoder from: {settings.LABEL_ENCODER_PATH}")
        label_encoder = joblib.load(settings.LABEL_ENCODER_PATH)

        input_size = len(vectorizer.get_feature_names_out())
        num_classes = len(label_encoder.classes_)

        logger.info("Initializing TextClassifier model structure")
        model = TextClassifier(input_size, num_classes)

        logger.info(f"Loading model weights from: {settings.MODEL_PATH}")
        # Use map_location='cpu' for cross-environment compatibility
        state_dict = torch.load(settings.MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()

        logger.info("Model artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {str(e)}", exc_info=True)
        raise RuntimeError(f"Model initialization failed: {str(e)}")

# Attempt eager load at import, but fallback gracefully if training hasn't occurred yet
try:
    load_model_artifacts()
except Exception as e:
    logger.warning(
        f"Eager model load failed: {str(e)}. API will try loading again on first request."
    )

def predict_category(text: str):
    """
    Vectorizes input text and runs inference through PyTorch.
    Returns category, confidence, and inference time.
    """
    # Ensure model is ready
    if model is None:
        load_model_artifacts()

    start = time.time()

    # Preprocess text and vectorize
    text_vec = vectorizer.transform([text]).toarray()
    
    # Zero-copy float tensor conversion
    text_tensor = torch.from_numpy(text_vec).float()

    with torch.no_grad():
        outputs = model(text_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    category = label_encoder.inverse_transform([predicted.item()])[0]
    
    end = time.time()

    return {
        "category": category,
        "confidence": round(confidence.item(), 4),
        "inference_time_seconds": round(end - start, 5)
    }