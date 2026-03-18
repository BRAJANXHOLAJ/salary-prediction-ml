# Salary Prediction using Linear Regression

## Overview

This project demonstrates a simple Machine Learning pipeline that predicts an employee's **salary based on years of experience** using **Linear Regression**.

The goal of the project is to practice the core steps of a machine learning workflow including data exploration, model training, evaluation, visualization, and saving the trained model for later use.

---

## Dataset

The dataset contains **30 observations** with two variables:

| Column          | Description                        |
| YearsExperience | Number of years of work experience |
| Salary          | Employee salary                    |

Example:

```
YearsExperience,Salary
1.1,39343
1.3,46205
2.0,43525
```

---

## Technologies Used

* Python
* pandas
* numpy
* matplotlib
* scikit-learn
* joblib

---

## Machine Learning Workflow

The project follows a typical ML pipeline:

1. **Load Dataset**
   Import data using pandas.

2. **Exploratory Data Analysis**
   Inspect dataset structure using `.head()`, `.info()`, and `.describe()`.

3. **Feature Selection**

```
X = YearsExperience
y = Salary
```

4. **Train/Test Split**

```
80% training
20% testing
```

5. **Model Training**

A **Linear Regression** model is trained to learn the relationship:

```
Salary = a × Experience + b
```

6. **Model Evaluation**

Metrics used:

* RMSE (Root Mean Squared Error)
* R² Score

Results:

```
RMSE ≈ 7059
R² ≈ 0.90
```

This means the model explains about **90% of the salary variation**.

7. **Visualization**

A regression plot shows:

* training data points
* regression line

8. **Prediction Example**

Example prediction for **6.5 years of experience**:

```
Predicted Salary ≈ $86,576
```

---

## Saving the Model

The trained model is saved using **joblib**:

```
salary_model.pkl
```

This allows the model to be **loaded later without retraining**.

Example usage:

```python
import joblib
import pandas as pd

model = joblib.load("salary_model.pkl")

data = pd.DataFrame([[6.5]], columns=["YearsExperience"])

prediction = model.predict(data)

print(prediction)
```

---

## Project Structure

```
salary-prediction-ml
│
├── dataset.csv
├── salary_prediction.py
├── predict_salary.py
├── salary_model.pkl
├── requirements.txt
└── README.md
```

---

## How to Run the Project

Install dependencies:

```
pip install -r requirements.txt
```

Run training script:

```
python salary_prediction.py
```

Run prediction script:

```
python predict_salary.py
```

---

## Learning Goals

This project demonstrates key machine learning concepts:

* regression modeling
* train/test splitting
* model evaluation
* data visualization
* saving trained models for reuse

---

## Future Improvements

Possible improvements:

* larger dataset
* additional features (education, location, job role)
* comparing multiple ML models
* building an API for predictions
