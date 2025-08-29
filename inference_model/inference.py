import joblib

# Load saved model + vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# User query
query = ["I enjoy sunny days"]

# Convert to vector
X_query = vectorizer.transform(query)

# Predict
prediction = model.predict(X_query)
print("Prediction:", prediction[0])
