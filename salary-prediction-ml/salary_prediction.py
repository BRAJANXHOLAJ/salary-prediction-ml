import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading the dataset
data = pd.read_csv("dataset.csv")
# Show first rows
print("First rows of dataset:")
print(data.head())


print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# Define features (X) and target (y)

X = data[["YearsExperience"]]
y = data["Salary"]

print("\nFeature variable (X):")
print(X.head())

print("\nTarget variable (y):")
print(y.head())


# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining feature set (X_train):")
print(X_train.head())

print("\nTesting feature set (X_test):")
print(X_test.head())


# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining feature set (X_train):")
print(X_train.head())

print("\nTesting feature set (X_test):")
print(X_test.head())


# Create Linear Regression model
model = LinearRegression()

# Train the model using the training dataset
model.fit(X_train, y_train)

print("\nModel Parameters:")

print("Slope (coefficient):", model.coef_)
print("Intercept:", model.intercept_)



# Make predictions using the test set
y_pred = model.predict(X_test)

print("\nPredicted salaries for test data:")
print(y_pred)


# Evaluate model performance

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("RMSE:", rmse)
print("R2 Score:", r2)


# Predict salary for a new employee

years_experience = pd.DataFrame([[6.5]], columns=["YearsExperience"])

predicted_salary = model.predict(years_experience)

print("\nPrediction for new employee:")
print("Years of Experience:", years_experience.iloc[0, 0])
print("Predicted Salary:", predicted_salary[0])


# Save trained model
joblib.dump(model, "salary_model.pkl")

print("\nModel saved successfully as salary_model.pkl")


# Visualize the regression line

plt.scatter(X_train, y_train, color="blue", label="Training Data")

plt.plot(X_train, model.predict(X_train), color="red", label="Regression Line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience (Linear Regression)")

plt.legend()
plt.savefig("regression_plot.png")
plt.show()



