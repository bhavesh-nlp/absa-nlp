from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictRequest, PredictResponse
from src.inference.aspect_extractor import AspectExtractor


app = FastAPI(
    title="ABSA API",
    description="Aspect-Based Sentiment Analysis using BERT BIO-POL",
    version="1.0.0"
)

# Allow CORS (useful for frontend / testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once)
aspect_extractor = None


@app.on_event("startup")
def load_model():
    """
    Load model at application startup.
    """
    global aspect_extractor
    try:
        aspect_extractor = AspectExtractor()
        print("AspectExtractor loaded successfully.")
    except Exception as e:
        print("Model loading failed:", str(e))
        aspect_extractor = None


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    if aspect_extractor is None:
        return {"status": "unhealthy", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Run ABSA prediction on input text.
    """
    if aspect_extractor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model before starting the API."
        )

    aspects = aspect_extractor.predict(request.text)
    return {"aspects": aspects}
