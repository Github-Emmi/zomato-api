# Zomato ML Project Architecture
## Step-by-Step Implementation Guide

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Architecture](#2-data-architecture)
3. [Phase 1: Environment Setup](#3-phase-1-environment-setup)
4. [Phase 2: Data Ingestion & Understanding](#4-phase-2-data-ingestion--understanding)
5. [Phase 3: Data Wrangling & Cleaning](#5-phase-3-data-wrangling--cleaning)
6. [Phase 4: Exploratory Data Analysis (EDA)](#6-phase-4-exploratory-data-analysis-eda)
7. [Phase 5: Hypothesis Testing](#7-phase-5-hypothesis-testing)
8. [Phase 6: Feature Engineering](#8-phase-6-feature-engineering)
9. [Phase 7: NLP & Sentiment Analysis](#9-phase-7-nlp--sentiment-analysis)
10. [Phase 8: Restaurant Clustering](#10-phase-8-restaurant-clustering)
11. [Phase 9: ML Model Implementation](#11-phase-9-ml-model-implementation)
12. [Phase 10: Model Evaluation & Selection](#12-phase-10-model-evaluation--selection)
13. [Phase 11: Model Deployment & Future Work](#13-phase-11-model-deployment--future-work)
14. [Deliverables Checklist](#14-deliverables-checklist)

---

## 1. Project Overview

### 1.1 Business Context
Zomato is an Indian restaurant aggregator and food delivery platform. This project analyzes restaurant data from the Gachibowli area of Hyderabad to:
- **Sentiment Analysis**: Analyze customer reviews to understand sentiment patterns
- **Restaurant Clustering**: Segment restaurants into meaningful clusters
- **Critic Identification**: Identify influential reviewers/critics
- **Business Insights**: Provide actionable recommendations for customers and the company

### 1.2 Problem Statements

| Task | Description | ML Type |
|------|-------------|---------|
| Sentiment Analysis | Classify reviews as Positive/Negative/Neutral | Classification (NLP) |
| Rating Prediction | Predict restaurant rating from reviews | Regression |
| Restaurant Clustering | Segment restaurants by features | Unsupervised (Clustering) |
| Reviewer Influence | Identify critics by metadata | Feature Analysis |

### 1.3 Data Sources

```
├── Zomato Restaurant names and Metadata.csv  (107 restaurants)
│   ├── Name, Links, Cost, Collections, Cuisines, Timings
│
├── Zomato Restaurant reviews.csv  (26,766 reviews)
│   ├── Restaurant, Reviewer, Review, Rating, Metadata, Time, Pictures
```

---

## 2. Data Architecture

### 2.1 Data Flow Diagram

```
┌─────────────────────┐     ┌─────────────────────┐
│   Restaurant        │     │   Reviews           │
│   Metadata.csv      │     │   Data.csv          │
└─────────┬───────────┘     └──────────┬──────────┘
          │                            │
          └────────────┬───────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │   Data Merging         │
          │   (on Restaurant Name) │
          └───────────┬────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Structured     │     │  Unstructured   │
│  Features       │     │  (Text Reviews) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Clustering     │     │  NLP Pipeline   │
│  (K-Means,      │     │  (Sentiment,    │
│   Hierarchical) │     │   Vectorization)│
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Combined ML Models │
          │  (Classification/   │
          │   Regression)       │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Model Evaluation   │
          │  & Selection        │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Deployment         │
          │  (pickle/joblib)    │
          └─────────────────────┘
```

### 2.2 Feature Categories

| Category | Features | Type |
|----------|----------|------|
| Restaurant Info | Name, Cost, Cuisines, Collections | Categorical/Numerical |
| Review Text | Review content | Text (NLP) |
| Review Metadata | Rating, Pictures, Time | Numerical/DateTime |
| Reviewer Info | Name, Followers, Review Count | Numerical |

---

## 3. Phase 1: Environment Setup

### 3.1 Virtual Environment Creation
```bash
# Navigate to project directory
cd "/Users/emmidev/Desktop/Zomato Project"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3.2 Project Structure
```
Zomato Project/
├── venv/                              # Virtual Environment
├── data/
│   ├── Zomato Restaurant names and Metadata.csv
│   └── Zomato Restaurant reviews.csv
├── models/                            # Saved models (to be created)
├── images/                            # Visualizations
├── ML_Submission_Template.ipynb       # Main working notebook
├── ML_PROJECT_ARCHITECTURE.md         # This document
└── requirements.txt                   # Dependencies
```

### 3.3 Jupyter Kernel Setup
```python
# Register virtual environment as Jupyter kernel
python -m ipykernel install --user --name=zomato_ml --display-name="Zomato ML"
```

---

## 4. Phase 2: Data Ingestion & Understanding

### 4.1 Import Libraries
```python
# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# NLP
import nltk
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score

# Statistical Tests
from scipy import stats
import statsmodels.api as sm

# Model Persistence
import joblib
import pickle

# Utilities
import warnings
import re
from collections import Counter
warnings.filterwarnings('ignore')
```

### 4.2 Load Datasets
```python
# Load metadata
metadata_df = pd.read_csv('Zomato Restaurant names and Metadata.csv')

# Load reviews
reviews_df = pd.read_csv('Zomato Restaurant reviews.csv')
```

### 4.3 Initial Data Exploration
```python
# Shape, Info, Head, Tail
print(f"Metadata Shape: {metadata_df.shape}")
print(f"Reviews Shape: {reviews_df.shape}")

# Display first/last rows
metadata_df.head()
reviews_df.head()

# Data types
metadata_df.info()
reviews_df.info()

# Statistical summary
metadata_df.describe()
reviews_df.describe()
```

### 4.4 Data Quality Assessment
- **Duplicates**: Check for duplicate rows
- **Missing Values**: Identify and count null values
- **Data Types**: Verify correct data types
- **Unique Values**: Count unique values per column

---

## 5. Phase 3: Data Wrangling & Cleaning

### 5.1 Metadata Cleaning Pipeline

| Step | Action | Code |
|------|--------|------|
| 1 | Remove duplicates | `df.drop_duplicates()` |
| 2 | Parse Cost (remove commas) | `df['Cost'] = df['Cost'].str.replace(',', '').astype(int)` |
| 3 | Split Cuisines into list | `df['Cuisines_List'] = df['Cuisines'].str.split(', ')` |
| 4 | Count Cuisines | `df['Cuisine_Count'] = df['Cuisines_List'].apply(len)` |
| 5 | Parse Collections | `df['Collection_List'] = df['Collections'].str.split(', ')` |
| 6 | Extract Operating Hours | Parse Timings column |

### 5.2 Reviews Cleaning Pipeline

| Step | Action | Code |
|------|--------|------|
| 1 | Remove duplicates | `df.drop_duplicates()` |
| 2 | Handle missing reviews | Drop or impute |
| 3 | Convert Rating to numeric | `pd.to_numeric(df['Rating'], errors='coerce')` |
| 4 | Parse Metadata | Extract followers, review count |
| 5 | Convert Time | `pd.to_datetime(df['Time'])` |
| 6 | Clean Pictures | Convert to integer |

### 5.3 Data Merging
```python
# Merge reviews with metadata
merged_df = reviews_df.merge(
    metadata_df, 
    left_on='Restaurant', 
    right_on='Name', 
    how='left'
)
```

---

## 6. Phase 4: Exploratory Data Analysis (EDA)

### 6.1 Visualization Requirements (15+ Charts)

#### Univariate Analysis (5+ charts)
| # | Chart | Purpose |
|---|-------|---------|
| 1 | Rating Distribution | Histogram of ratings |
| 2 | Cost Distribution | Box plot of costs |
| 3 | Cuisine Frequency | Bar chart of popular cuisines |
| 4 | Collection Distribution | Pie chart of collections |
| 5 | Reviews per Restaurant | Bar chart |
| 6 | Review Length Distribution | Histogram |

#### Bivariate Analysis (5+ charts)
| # | Chart | Purpose |
|---|-------|---------|
| 7 | Cost vs Rating | Scatter plot |
| 8 | Cuisines vs Avg Rating | Grouped bar chart |
| 9 | Collection vs Avg Cost | Box plot |
| 10 | Review Count vs Rating | Scatter plot |
| 11 | Reviewer Followers vs Rating | Correlation |

#### Multivariate Analysis (5+ charts)
| # | Chart | Purpose |
|---|-------|---------|
| 12 | Heatmap | Correlation matrix |
| 13 | Pair Plot | Numerical features |
| 14 | 3D Scatter | Cost, Rating, Review Count |
| 15 | Cluster Visualization | PCA/t-SNE |
| 16 | Word Cloud | Top words in reviews |
| 17 | Sentiment by Restaurant | Stacked bar |

### 6.2 Business Insights to Extract
1. **Top-rated restaurants** by average rating
2. **Most expensive vs best value** restaurants
3. **Popular cuisines** in Gachibowli area
4. **Peak review times** - temporal patterns
5. **Reviewer influence** - critics with many followers

---

## 7. Phase 5: Hypothesis Testing

### 7.1 Hypothesis Statements (3 Required)

#### Hypothesis 1: Cost & Rating Relationship
- **H₀**: There is no significant correlation between restaurant cost and rating
- **H₁**: Higher-cost restaurants receive significantly higher ratings
- **Test**: Pearson/Spearman Correlation, t-test

#### Hypothesis 2: Cuisine Type Impact
- **H₀**: Different cuisine types have equal average ratings
- **H₁**: At least one cuisine type has significantly different ratings
- **Test**: ANOVA (One-way)

#### Hypothesis 3: Review Pictures & Rating
- **H₀**: Reviews with pictures have the same rating distribution as those without
- **H₁**: Reviews with pictures have higher ratings
- **Test**: Mann-Whitney U Test / t-test

### 7.2 Statistical Test Implementation
```python
from scipy.stats import ttest_ind, pearsonr, spearmanr, f_oneway, mannwhitneyu

# Significance level
alpha = 0.05

# Example: Hypothesis 1
correlation, p_value = pearsonr(df['Cost'], df['Rating'])
if p_value < alpha:
    print("Reject H0: Significant correlation exists")
else:
    print("Fail to reject H0: No significant correlation")
```

---

## 8. Phase 6: Feature Engineering

### 8.1 Feature Creation Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. NUMERICAL FEATURES                                  │
│     ├── Cost (cleaned, normalized)                      │
│     ├── Rating (target/feature)                         │
│     ├── Cuisine_Count                                   │
│     ├── Collection_Count                                │
│     ├── Review_Length                                   │
│     ├── Reviewer_Followers                              │
│     ├── Reviewer_Review_Count                           │
│     └── Pictures_Count                                  │
│                                                         │
│  2. CATEGORICAL FEATURES                                │
│     ├── Cuisines (one-hot encoded)                      │
│     ├── Collections (one-hot/binary)                    │
│     ├── Has_Pictures (binary)                           │
│     ├── Time_of_Day (Morning/Afternoon/Evening/Night)   │
│     └── Day_of_Week                                     │
│                                                         │
│  3. TEXT-DERIVED FEATURES                               │
│     ├── Sentiment_Score (TextBlob polarity)             │
│     ├── Sentiment_Category (Pos/Neg/Neutral)            │
│     ├── Subjectivity_Score                              │
│     ├── Word_Count                                      │
│     └── TF-IDF Vectors                                  │
│                                                         │
│  4. AGGREGATED FEATURES (Restaurant-level)              │
│     ├── Avg_Rating                                      │
│     ├── Total_Reviews                                   │
│     ├── Avg_Sentiment                                   │
│     ├── Review_Variance                                 │
│     └── High_Follower_Review_Ratio                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 8.2 Encoding Strategies

| Feature Type | Encoding Method |
|--------------|-----------------|
| Binary Categories | Label Encoding (0/1) |
| Multi-class Ordinal | Label Encoding |
| Multi-class Nominal | One-Hot Encoding |
| High Cardinality | Target Encoding / Hashing |
| Text | TF-IDF / Word2Vec / Count Vectorizer |

### 8.3 Scaling & Normalization
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard Scaling for most features
scaler = StandardScaler()
numerical_features = ['Cost', 'Review_Length', 'Sentiment_Score']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# MinMax for bounded features
minmax = MinMaxScaler()
df['Rating_Normalized'] = minmax.fit_transform(df[['Rating']])
```

---

## 9. Phase 7: NLP & Sentiment Analysis

### 9.1 Text Preprocessing Pipeline

```
┌────────────────────────────────────────────────┐
│              NLP PREPROCESSING                 │
├────────────────────────────────────────────────┤
│                                                │
│  1. LOWERCASE CONVERSION                       │
│     "Great Food!" → "great food!"              │
│                                                │
│  2. REMOVE SPECIAL CHARACTERS/PUNCTUATION      │
│     "great food!" → "great food"               │
│                                                │
│  3. TOKENIZATION                               │
│     "great food" → ["great", "food"]           │
│                                                │
│  4. STOPWORD REMOVAL                           │
│     ["the", "is", "a"] removed                 │
│                                                │
│  5. LEMMATIZATION                              │
│     "running" → "run"                          │
│     "better" → "good"                          │
│                                                │
│  6. VECTORIZATION                              │
│     ├── TF-IDF Vectorizer                      │
│     ├── Count Vectorizer                       │
│     └── Word2Vec (optional)                    │
│                                                │
└────────────────────────────────────────────────┘
```

### 9.2 Sentiment Analysis Implementation
```python
from textblob import TextBlob

def get_sentiment(text):
    """Calculate sentiment polarity and subjectivity"""
    if pd.isna(text):
        return 0, 0, 'Neutral'
    
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity  # -1 to 1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1
    
    # Categorize
    if polarity > 0.1:
        category = 'Positive'
    elif polarity < -0.1:
        category = 'Negative'
    else:
        category = 'Neutral'
    
    return polarity, subjectivity, category

# Apply to reviews
df[['Polarity', 'Subjectivity', 'Sentiment']] = df['Review'].apply(
    lambda x: pd.Series(get_sentiment(x))
)
```

### 9.3 Word Cloud Generation
```python
from wordcloud import WordCloud

# Positive reviews word cloud
positive_reviews = ' '.join(df[df['Sentiment'] == 'Positive']['Review_Clean'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Positive Reviews')
plt.savefig('images/positive_wordcloud.png')
```

---

## 10. Phase 8: Restaurant Clustering

### 10.1 Clustering Feature Selection
```python
clustering_features = [
    'Cost_Normalized',
    'Avg_Rating',
    'Total_Reviews',
    'Cuisine_Count',
    'Avg_Sentiment',
    'Collection_Count'
]
```

### 10.2 Clustering Algorithms

#### Algorithm 1: K-Means
```python
from sklearn.cluster import KMeans

# Elbow Method for optimal K
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
```

#### Algorithm 2: Hierarchical Clustering
```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Compute linkage
linked = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='lastp', p=10)
plt.title('Hierarchical Clustering Dendrogram')
```

#### Algorithm 3: DBSCAN (Optional)
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)
```

### 10.3 Cluster Evaluation
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Silhouette Score
silhouette = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette:.3f}")

# Calinski-Harabasz Score
ch_score = calinski_harabasz_score(X_scaled, clusters)
print(f"Calinski-Harabasz Score: {ch_score:.3f}")
```

### 10.4 Expected Cluster Segments

| Cluster | Profile | Characteristics |
|---------|---------|-----------------|
| 0 | Budget-Friendly | Low cost, varied ratings, high volume |
| 1 | Premium Dining | High cost, high ratings, multiple cuisines |
| 2 | Popular Hotspots | Medium cost, very high reviews, positive sentiment |
| 3 | Specialized | Single cuisine, niche audience |
| 4 | Hidden Gems | Low reviews, but high ratings |

---

## 11. Phase 9: ML Model Implementation

### 11.1 Problem Formulation

| Task | Type | Target | Features |
|------|------|--------|----------|
| Sentiment Classification | Classification | Sentiment (Pos/Neg/Neu) | TF-IDF vectors |
| Rating Prediction | Regression | Rating (1-5) | All features |
| Restaurant Success | Classification | High/Low Rating | Restaurant features |

### 11.2 Train-Test Split
```python
from sklearn.model_selection import train_test_split

X = df[feature_columns]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 11.3 Model Pipeline (3 Models Required)

#### Model 1: Logistic Regression (Baseline)
```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
```

#### Model 2: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
```

#### Model 3: Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
```

### 11.4 Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.3f}")
```

---

## 12. Phase 10: Model Evaluation & Selection

### 12.1 Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
```

### 12.2 Confusion Matrix Visualization
```python
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'images/confusion_matrix_{title.lower().replace(" ", "_")}.png')
```

### 12.3 Model Comparison
```python
# Compare all models
results = {
    'Logistic Regression': evaluate_model(y_test, lr_pred, 'Logistic Regression'),
    'Random Forest': evaluate_model(y_test, rf_pred, 'Random Forest'),
    'Gradient Boosting': evaluate_model(y_test, gb_pred, 'Gradient Boosting')
}

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T
comparison_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('images/model_comparison.png')
```

### 12.4 ROC Curve & AUC
```python
from sklearn.metrics import roc_curve, auc

# For binary classification
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('images/roc_curve.png')
```

---

## 13. Phase 11: Model Deployment & Future Work

### 13.1 Save Best Model
```python
import joblib
import pickle

# Create models directory
import os
os.makedirs('models', exist_ok=True)

# Save with joblib (recommended for sklearn)
joblib.dump(best_model, 'models/zomato_sentiment_model.joblib')
joblib.dump(scaler, 'models/feature_scaler.joblib')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')

# Alternative: Save with pickle
with open('models/zomato_sentiment_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
```

### 13.2 Load Model for Inference
```python
# Load model
loaded_model = joblib.load('models/zomato_sentiment_model.joblib')
scaler = joblib.load('models/feature_scaler.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

# Make prediction
def predict_sentiment(review_text):
    # Preprocess
    cleaned = preprocess_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = loaded_model.predict(vectorized)
    probability = loaded_model.predict_proba(vectorized)
    return prediction[0], probability[0]
```

### 13.3 Future Work Recommendations
1. **Deep Learning**: Implement LSTM/BERT for better sentiment accuracy
2. **Real-time Pipeline**: Create streaming data pipeline
3. **API Development**: Flask/FastAPI for model serving
4. **Dashboard**: Interactive Power BI/Tableau dashboard
5. **More Data**: Expand to other Hyderabad areas
6. **Recommendation Engine**: Collaborative filtering for restaurant suggestions

---

## 14. Deliverables Checklist

### 14.1 Notebook Rubrics

| Section | Requirement | Status |
|---------|-------------|--------|
| Summary | Technical documentation | ⬜ |
| EDA | ≥15 visualizations | ⬜ |
| Data Cleaning | Missing values, duplicates | ⬜ |
| Feature Engineering | Text preprocessing, encoding | ⬜ |
| Hypothesis Testing | 3 statistical tests | ⬜ |
| ML Models | 3 different algorithms | ⬜ |
| Evaluation | Metrics comparison | ⬜ |
| Conclusion | Summary of findings | ⬜ |
| Code Comments | Properly documented | ⬜ |
| Modularity | Functions, clean structure | ⬜ |

### 14.2 Output Files
```
Zomato Project/
├── ML_Submission_Template.ipynb    # Completed notebook
├── models/
│   ├── zomato_sentiment_model.joblib
│   ├── feature_scaler.joblib
│   └── tfidf_vectorizer.joblib
├── images/
│   ├── rating_distribution.png
│   ├── cost_distribution.png
│   ├── cuisine_frequency.png
│   ├── correlation_heatmap.png
│   ├── positive_wordcloud.png
│   ├── negative_wordcloud.png
│   ├── cluster_visualization.png
│   ├── confusion_matrix_*.png
│   ├── model_comparison.png
│   └── roc_curve.png
└── reports/
    └── executive_summary.pdf
```

---

## 📊 Quick Reference: Key Python Snippets

### Load Data
```python
metadata = pd.read_csv('Zomato Restaurant names and Metadata.csv')
reviews = pd.read_csv('Zomato Restaurant reviews.csv')
```

### Basic EDA
```python
df.shape, df.info(), df.describe()
df.isnull().sum(), df.duplicated().sum()
```

### Sentiment
```python
from textblob import TextBlob
df['sentiment'] = df['Review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
```

### Clustering
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5).fit(X_scaled)
```

### Classification
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X_train, y_train)
accuracy_score(y_test, model.predict(X_test))
```

---

**Document Version**: 1.0  
**Last Updated**: March 9, 2026  
**Author**: Professional ML Engineer  
**Project**: Zomato Restaurant Analysis - Hyderabad (Gachibowli)
