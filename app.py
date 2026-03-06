from flask import Flask, render_template, request
import sklearn
import numpy as np
import pickle

app = Flask(__name__)

# Load trained custom model
with open("credit_model.pkl", "rb") as f:
    data = pickle.load(f)

weights = data["weights"]
bias = data["bias"]
scaler = data["scaler"]

# Helper functions (same as Colab)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_custom(X, weights, bias):
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)
    return (y_pred >= 0.5).astype(int)

def predict_proba_custom(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["gender"]),
            float(request.form["own_car"]),
            float(request.form["own_property"]),
            float(request.form["employed"]),
            float(request.form["Num_Children"]),
            float(request.form["Num_Family"]),
            float(request.form["Account_Length"]),
            float(request.form["Total_Income"]),
            float(request.form["Age"]),
            float(request.form["Income_Type"]),
            float(request.form["Education_Type"]),
            float(request.form["family_status"]),
            float(request.form["Housing_Type"]),
            float(request.form["occupation_type"]),
        ]

        final_features = scaler.transform([features])

        prediction = predict_custom(final_features, weights, bias)[0]
        proba = predict_proba_custom(final_features, weights, bias)[0]

        confidence = round(proba * 100, 2)

        result = (
            f"Approved ✅ (Confidence: {confidence}%)"
            if prediction == 1
            else f"Rejected ❌ (Confidence: {100 - confidence}%)"
        )

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        print("Error:", e)
        return render_template("index.html", prediction_text="Invalid Input ❌")

if __name__ == "__main__":
    app.run()