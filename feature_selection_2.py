from sklearn.preprocessing import StandardScalar

scaler = StandardScalar()

"""
1)Building a diabetes classifier
You'll be using the Pima Indians diabetes dataset to predict whether a person has diabetes using logistic regression. 
There are 8 features and one target in this dataset. The data has been split into a training and test set and pre-loaded for you as X_train, y_train, X_test, and y_test.

A StandardScaler() instance has been predefined as scaler and a LogisticRegression() one as lr

"""


# Fit the scaler on the training features and transform these in one go
X_train_std = scaler.fit_transform(X_train)

# Fit the logistic regression model on the scaled training data
lr.fit(X_train_std, y_train)

# Scale the test features
X_test_std = scaler.transform(X_test)

# Predict diabetes presence on the scaled test set
y_pred = lr.predict(X_test_std)

# Prints accuracy metrics and feature coefficients
print("{0:.1%} accuracy on test set.".format(accuracy_score(y_test, y_pred)))
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))


"""
79.6% accuracy on test set.
{'bmi': 0.38, 'insulin': 0.19, 'glucose': 1.23, 'diastolic': 0.03, 'family': 0.34, 'age': 0.34, 'triceps': 0.24, 'pregnant': 0.04}

Great! We get almost 80% accuracy on the test set. Take a look at the differences in model coefficients for the different features.
"""



"""
2) Manual Recursive Feature Elimination 

Now that we've created a diabetes classifier, let's see if we can reduce the number of features without hurting the 
model accuracy too much.

On the second line of code the features are selected from the original dataframe. Adjust this selection.

A StandardScaler() instance has been predefined as scaler and a LogisticRegression() one as lr.

All necessary functions and packages have been pre-loaded too.


"""

# Only keep the feature with the highest coefficient
X = diabetes_df[['glucose', 'triceps', 'bmi', 'family', 'age']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model to the data
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print("{0:.1%} accuracy on test set.".format(acc))
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

"""
Removed the features, one after another based on the lowest coefficient or the feature importance. 
"""