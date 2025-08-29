import joblib

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Ask user for input
user_input = input("Enter a sentence to analyze: ")

# Transform the input using the vectorizer
X = vectorizer.transform([user_input])

# Get prediction and probability
prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

print(f"\nPrediction: {prediction}")
print(f"Confidence: {max(probabilities) * 100:.2f}%")