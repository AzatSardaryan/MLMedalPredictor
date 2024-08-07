import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import numpy as np

# Load and preprocess the data
teams = pd.read_csv("teams.csv")
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

# Check for empty values and then drop them
teams = teams.dropna()

# Split the data into training and test sets
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

# Define predictors and target
predictors = ["athletes", "prev_medals"]
target = "medals"

# Apply polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
train_poly = poly.fit_transform(train[predictors])
test_poly = poly.transform(test[predictors])

# Standardize the features
scaler = StandardScaler()
train_poly = scaler.fit_transform(train_poly)
test_poly = scaler.transform(test_poly)

# Train the model
reg = LinearRegression()
reg.fit(train_poly, train[target])

# Make predictions
predictions = reg.predict(test_poly)

# Add predictions to the test data frame
test["predictions"] = predictions

# Index test data and find all predictions less than 0 and set the predictions value to 0
test.loc[test["predictions"] < 0, "predictions"] = 0

# Round the rest of the predictions
test["predictions"] = test["predictions"].round()

# Calculate mean absolute error
error = mean_absolute_error(test["medals"], test["predictions"])
print(f"Mean Absolute Error: {error}")

# Check the standard deviation of medals
print(teams.describe()["medals"])

# Check how well we did in predicting scores for specific teams
print(test[test["team"] == "USA"])
print(test[test["team"] == "IND"])

# Calculate the absolute errors
errors = (test["medals"] - test["predictions"]).abs()

# Calculate mean error by team
error_by_team = errors.groupby(test["team"]).mean()

# Calculate average medals by team
medals_by_team = test["medals"].groupby(test["team"]).mean()

# Calculate error ratio
error_ratio = error_by_team / medals_by_team

# Remove NaN and infinity values
error_ratio = error_ratio[~pd.isnull(error_ratio)]
error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio = error_ratio.sort_values()

# Plot error ratio histogram
error_ratio.plot.hist()
plt.xlabel('Error Ratio')
plt.ylabel('Frequency')
plt.title('Distribution of Error Ratios')
plt.show()

print(error_ratio)
