import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

teams = pd.read_csv("teams.csv")

# get the cols we want to use to test our hypothesis
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

# Look at linear plots to see if a relationship exists
sns.lmplot(x="athletes", y="medals", data=teams,fit_reg=True,ci=None)
sns.lmplot(x="age", y="medals", data=teams,fit_reg=True,ci=None)


teams.plot.hist(y="medals")

#check for empty values and then drop them
teams = teams.dropna()


#we want our training set to be on previous years and our test set to be on more recent years. 
# this is because if we want to predict 2024 results we will only have past years to go off.
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

# check the training data - we are looking for a 80 - 20 split (training - test)

reg = LinearRegression()

#Use atheletes and prev medals to predict medals column
predictors = ["athletes", "prev_medals"]
target = "medals"


reg.fit(train[predictors], train["medals"])

LinearRegression()

predictions = reg.predict(test[predictors])

#Add our predictions to the test data frame
test["predictions"] = predictions

#Index test data and find all predictions less than 0 and set the predictions value to 0
test.loc[test["predictions"] < 0, "predictions"] = 0

#Round the rest of the predictions
test["predictions"] = test["predictions"].round()

error = mean_absolute_error(test["medals"], test["predictions"])

print(error)

#check the standard diviation - if our mean absolute error is above the std then something is wrong
print(teams.describe()["medals"])

#Check how well we did in predicting scores
print(test[test["team"] == "USA"])
print(test[test["team"] == "IND"])


#How far off we were in our predictions
errors = (test["medals"] - test["predictions"]).abs()


error_by_team = errors.groupby(test["team"]).mean()

#How many medals each team earned on average
medals_by_team = test["medals"].groupby(test["team"]).mean()

error_ratio = error_by_team / medals_by_team

#If it is null remove the values
error_ratio = error_ratio[~pd.isnull(error_ratio)]

#Clean up the infinity  values from dividing by 0
error_ratio = error_ratio[np.isfinite(error_ratio)]

error_ratio = error_ratio.sort_values()

error_ratio.plot.hist()

print(error_ratio)

#print(test)
plt.show()