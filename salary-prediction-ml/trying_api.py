import pandas as pd
import joblib

# Load saved model
model = joblib.load("salary_model.pkl")

# Example prediction
input_data = pd.DataFrame([[7]], columns=["YearsExperience"])

prediction = model.predict(input_data)

print("Predicted salary:", prediction[0])