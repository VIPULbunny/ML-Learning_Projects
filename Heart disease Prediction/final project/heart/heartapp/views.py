from django.shortcuts import render
import joblib
import numpy as np
import os
import pandas as pd

model_path = os.path.join(os.path.dirname(__file__), "logistic.pkl")  
model = joblib.load(model_path)

def predict(request):
    if request.method == "POST":
        features = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", 
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

        try:
            input_values = [float(request.POST.get(feature, 0)) for feature in features]

            input_df = pd.DataFrame([input_values], columns=features)

            print("Input DataFrame for Prediction:\n", input_df)

            prediction = model.predict(input_df)[0]

            result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

            return render(request, "heartapp/predict.html", {"result": result})

        except Exception as e:
            print("Error Occurred:", str(e))
            return render(request, "heartapp/predict.html", {"error": f"Error: {str(e)}"})

    return render(request, "heartapp/predict.html")