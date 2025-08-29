# 1. Install dependencies (if not already)
# pip install scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# --- Step 1: Training ---
# Example dataset
texts = [
    "I love pizza", 
    "I hate pizza", 
    "The weather is great", 
    "The weather is terrible"
]
labels = ["positive", "negative", "positive", "negative"]

# Convert text → numerical vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train a simple classifier
model = MultinomialNB()
model.fit(X, labels)

# Save model + vectorizer for later inference
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained and saved")
