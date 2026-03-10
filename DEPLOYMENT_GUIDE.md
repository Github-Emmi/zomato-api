# Zomato Restaurant Sentiment Classifier - Deployment Guide

## 🎯 Deployment Stack
- **API Framework:** FastAPI (Modern & Async)
- **Containerization:** Docker
- **Cloud Platform:** Render

---

## 📋 Step-by-Step Deployment Process

### Phase 1: Prepare Project Files

#### Step 1.1: Create Project Directory
```bash
mkdir zomato-api
cd zomato-api
```

#### Step 1.2: Copy Model Files
Copy these files from your notebook output to the project directory:
```
zomato-api/
├── best_restaurant_classifier.pkl
├── feature_scaler.pkl
└── tfidf_vectorizer.pkl
```

---

### Phase 2: Create FastAPI Application

#### Step 2.1: Create `main.py`
```python
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
```

#### Step 2.2: Create `requirements.txt`
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.6.0
scikit-learn==1.4.0
numpy==2.0.0
python-multipart==0.0.9
```

---

### Phase 3: Docker Containerization

#### Step 3.1: Create `Dockerfile`
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY *.pkl .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 3.2: Create `.dockerignore`
```
__pycache__
*.pyc
*.pyo
.git
.gitignore
.env
*.md
.pytest_cache
venv/
```

#### Step 3.3: Test Docker Locally
```bash
# Build image
docker build -t zomato-classifier .

# Run container
docker run -p 8000:8000 zomato-classifier

# Test endpoint
curl http://localhost:8000/health
```

---

### Phase 4: Deploy to Render

#### Step 4.1: Prepare GitHub Repository
```bash
# Initialize git
git init

# Create .gitignore
echo "venv/
__pycache__/
*.pyc
.env
.DS_Store" > .gitignore

# Add files
git add .
git commit -m "Initial commit: Zomato API"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/zomato-api.git
git branch -M main
git push -u origin main
```

#### Step 4.2: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub account

#### Step 4.3: Create New Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Configure settings:

| Setting | Value |
|---------|-------|
| **Name** | `zomato-sentiment-api` |
| **Region** | Select nearest to your users |
| **Branch** | `main` |
| **Runtime** | `Docker` |
| **Instance Type** | `Free` (or paid for production) |

#### Step 4.4: Environment Variables (Optional)
Add in Render dashboard if needed:
```
PORT=8000
LOG_LEVEL=INFO
```

#### Step 4.5: Deploy
1. Click **"Create Web Service"**
2. Wait for build to complete (~5-10 minutes)
3. Your API will be live at: `https://zomato-sentiment-api.onrender.com`

---

### Phase 5: Verify Deployment

#### Step 5.1: Test Health Endpoint
```bash
curl https://zomato-sentiment-api.onrender.com/health
```

Expected response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "version": "1.0.0"
}
```

#### Step 5.2: Test Prediction Endpoint
```bash
curl -X POST https://zomato-sentiment-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "review": "Excellent biryani, will definitely visit again!",
    "cost": 600,
    "follower_count": 50,
    "has_pictures": 1
  }'
```

Expected response:
```json
{
    "prediction": "High",
    "confidence": 0.8523,
    "sentiment_score": 0.0234,
    "status": "success"
}
```

#### Step 5.3: Access API Documentation
Open in browser: `https://zomato-sentiment-api.onrender.com/docs`

---

## 📁 Final Project Structure
```
zomato-api/
├── main.py                           # FastAPI application
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker configuration
├── .dockerignore                     # Docker ignore rules
├── .gitignore                        # Git ignore rules
├── best_restaurant_classifier.pkl   # Trained model
├── feature_scaler.pkl               # Feature scaler
└── tfidf_vectorizer.pkl             # TF-IDF vectorizer
```

---

## ✅ Production Checklist

- [x] FastAPI with async endpoints
- [x] Input validation with Pydantic
- [x] Health check endpoint
- [x] CORS enabled for frontend access
- [x] Logging configured
- [x] Docker containerization
- [x] Render deployment with auto-deploy
- [ ] Add API key authentication (optional)
- [ ] Set up custom domain (optional)
- [ ] Configure monitoring alerts (optional)

---

## 🧪 Testing the Deployed API

### Python Client
```python
import requests

API_URL = "https://zomato-sentiment-api.onrender.com"

# Single prediction
response = requests.post(
    f"{API_URL}/predict",
    json={
        "review": "Terrible food, never coming back!",
        "cost": 400,
        "follower_count": 20,
        "has_pictures": 0
    }
)
print(response.json())

# Batch prediction
reviews = [
    {"review": "Amazing food!", "cost": 500},
    {"review": "Average experience", "cost": 300},
    {"review": "Worst restaurant ever", "cost": 200}
]
response = requests.post(f"{API_URL}/predict/batch", json=reviews)
print(response.json())
```

### JavaScript/Frontend
```javascript
const response = await fetch('https://zomato-sentiment-api.onrender.com/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        review: 'Great ambiance and tasty food!',
        cost: 700,
        follower_count: 100,
        has_pictures: 1
    })
});
const data = await response.json();
console.log(data.prediction, data.confidence);
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails on Render | Check `requirements.txt` versions match your local env |
| Model loading error | Verify `.pkl` files are committed to GitHub |
| Health check fails | Ensure port 8000 is exposed in Dockerfile |
| Slow cold starts | Upgrade to paid Render plan for always-on |

---

## 📞 Support

For deployment issues:
1. Check Render logs in dashboard
2. Test Docker locally first
3. Verify model file integrity: `python -c "import pickle; pickle.load(open('best_restaurant_classifier.pkl', 'rb'))"`

---

*Deployment Guide for Zomato Restaurant Sentiment Classifier*
*Stack: FastAPI + Docker + Render*
