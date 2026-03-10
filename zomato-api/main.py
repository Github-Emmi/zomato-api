from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import pickle
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model variables
model = None
scaler = None
tfidf = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    global model, scaler, tfidf
    logger.info("Loading ML models...")
    
    model = pickle.load(open('best_restaurant_classifier.pkl', 'rb'))
    scaler = pickle.load(open('feature_scaler.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    
    logger.info("✅ Models loaded successfully!")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Zomato Sentiment Classifier API",
    description="Predict restaurant review sentiment using ML",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ReviewInput(BaseModel):
    review: str = Field(..., min_length=1, description="Restaurant review text")
    cost: float = Field(default=500.0, ge=0, description="Cost for two people")
    follower_count: int = Field(default=0, ge=0, description="Reviewer follower count")
    has_pictures: int = Field(default=0, ge=0, le=1, description="1 if review has pictures")

    class Config:
        json_schema_extra = {
            "example": {
                "review": "Amazing biryani and excellent service!",
                "cost": 800,
                "follower_count": 150,
                "has_pictures": 1
            }
        }

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    sentiment_score: float
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    return {"message": "Zomato Sentiment Classifier API", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_sentiment(input_data: ReviewInput):
    """Predict sentiment category for a restaurant review"""
    try:
        # Process text with TF-IDF
        text_features = tfidf.transform([input_data.review]).toarray()
        avg_tfidf = float(np.mean(text_features))
        
        # Create feature vector
        features = np.array([[
            input_data.cost, 
            input_data.follower_count, 
            input_data.has_pictures, 
            avg_tfidf
        ]])
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(probabilities.max())
        
        categories = {0: 'Low', 1: 'Medium', 2: 'High'}
        
        logger.info(f"Prediction: {categories.get(prediction)} | Confidence: {confidence:.4f}")
        
        return PredictionOutput(
            prediction=categories.get(prediction, str(prediction)),
            confidence=round(confidence, 4),
            sentiment_score=round(avg_tfidf, 4),
            status="success"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(reviews: list[ReviewInput]):
    """Predict sentiment for multiple reviews"""
    results = []
    for review in reviews:
        result = await predict_sentiment(review)
        results.append(result)
    return {"predictions": results, "count": len(results)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
