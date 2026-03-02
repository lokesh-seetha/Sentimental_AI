import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# === 1. LOAD DATASET ===
# Path is relative to project root: D:\Gen_AI_Project\data\tripadvisor_hotel_reviews.csv
csv_path = os.path.join("data", "tripadvisor_hotel_reviews.csv")
df = pd.read_csv(csv_path)

# Kaggle TripAdvisor dataset usually has columns like: "Review", "Rating"
# If your column names differ, change them here.
TEXT_COL = "Review"
RATING_COL = "Rating"

# Keep only needed columns
df = df[[TEXT_COL, RATING_COL]].dropna()

# === 2. MAP RATINGS TO SENTIMENT LABELS ===
# You can tweak this mapping if your dataset is different.
def rating_to_sentiment(r):
    if r >= 4:
        return "positive"
    elif r <= 2:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df[RATING_COL].apply(rating_to_sentiment)

X = df[TEXT_COL].astype(str)
y = df["sentiment"]

# === 3. SPLIT TRAIN/TEST ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 4. BUILD PIPELINE: TF-IDF + LOGISTIC REGRESSION ===
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

# === 5. TRAIN MODEL ===
print("Training model on TripAdvisor dataset...")
pipeline.fit(X_train, y_train)

# === 6. EVALUATE ===
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === 7. SAVE MODEL PIPELINE ===
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "sentiment_pipeline.pkl")
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")
