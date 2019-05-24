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

######################


"""
3) Automatic Recursive Feature Elimination
Now let's automate this recursive process. Wrap a Recursive Feature Eliminator (RFE) 
around our logistic regression estimator and pass it the desired number of features.

All the necessary functions and packages have been pre-loaded and the features have been scaled for you.

"""
from sklearn.feature_selection import RFE

# Create the RFE with a LogisticRegression estimator and 3 features to select
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=3, verbose=1)

# Fits the eliminator to the data
rfe.fit(X_train, y_train)

# Print the features and their ranking (high = dropped early on)
print(dict(zip(X.columns, rfe.ranking_)))

# Print the features that are not eliminated
print(X.columns[rfe.support_])

# Calculates the test set accuracy
acc = accuracy_score(y_test, rfe.predict(X_test))
print("{0:.1%} accuracy on test set.".format(acc))


"""
Fitting estimator with 8 features.
Fitting estimator with 7 features.
Fitting estimator with 6 features.
Fitting estimator with 5 features.
Fitting estimator with 4 features.

{'bmi': 1, 'insulin': 4, 'glucose': 1, 'diastolic': 6, 'family': 2, 'age': 1, 'triceps': 3, 'pregnant': 5}

Index(['glucose', 'bmi', 'age'], dtype='object')

80.6% accuracy on test set.
"""
#########################

"""
4) Building Random Forest Classifier.

You'll again work on the Pima Indians dataset to predict whether an individual has diabetes. 
This time using a random forest classifier. You'll fit the model on the training data after performing the train-test split and consult the feature importance values.

The feature and target datasets have been pre-loaded for you as X and y. 
Same goes for the necessary packages and functions.

"""

# Perform a 75% training and 25% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fit the random forest model to the training data
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)

# Calculate the accuracy
acc = accuracy_score(y_test, rf.predict(X_test))

# Print the importances per feature
print(dict(zip(X.columns, rf.feature_importances_.round(2))))

# Print accuracy
print("{0:.1%} accuracy on test set.".format(acc))


"""
{'pregnant': 0.09, 'glucose': 0.21, 'triceps': 0.11, 'insulin': 0.13, 'family': 0.12, 'age': 0.16, 'diastolic': 0.08, 'bmi': 0.09}
77.6% accuracy on test set.

The random forest model gets 78% accuracy on the test set and 'glucose' is the most important feature (0.21)



"""


################

"""
5) Removing less important features from the data. 

"""

# Create a mask for features importances above the threshold
mask = rf.feature_importances_ > 0.15

# Apply the mask to the feature dataset X
reduced_X = X.loc[:,mask]

# prints out the selected column names
print(reduced_X.columns)

"""
Index(['glucose', 'age'], dtype='object')

But Removing less important features all at once, is not the right way. Since feature importance changes after each
iteration, that's why we need a RFE.
"""


#######################

"""
6) Recursive Feature Elimination with random forests
You'll wrap a Recursive Feature Eliminator around a random forest model to remove features step by step. 
This method is more conservative compared to selecting features after applying a single importance threshold. Since dropping one feature can influence the relative importances of the others.

You'll need these pre-loaded datasets: X, X_train, y_train.

Functions and classes that have been pre-loaded for you are: RandomForestClassifier(), RFE(), train_test_split().

"""

# Set the feature eliminator to remove 2 features on each step
rfe = RFE(estimator=RandomForestClassifier(random_state=0), n_features_to_select=2, step=2, verbose=1)

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print(reduced_X.columns)


"""
Fitting estimator with 8 features.
Fitting estimator with 7 features.
Fitting estimator with 6 features.
Fitting estimator with 5 features.
Fitting estimator with 4 features.
Fitting estimator with 3 features.
Out[2]: 
RFE(estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False),
  n_features_to_select=2, step=1, verbose=1)

Great! Compared to the quick and dirty single threshold method from the previous
exercise one of the selected features is different.

"""


################

"""
6) Creating a LASSO regressor

You'll be working on the numeric ANSUR body measurements dataset to predict a persons Body Mass Index (BMI) using the pre-imported Lasso() regressor. BMI is a metric derived from body height and weight but those two features have been removed from the dataset to give the model a challenge.

You'll standardize the data first using the StandardScaler() that has been instantiated for you as scaler to make sure all coefficients face a comparable regularizing force trying to bring them down.

All necessary functions and classes plus the input datasets X and y have been pre-loaded.

"""

# Set the test size to 30% to get a 70-30% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

# Fit the scaler on the training features and transform these in one go
X_train_std = scaler.fit_transform(X_train)

# Create the Lasso model
la = Lasso()

# Fit it to the standardized training data
la.fit(X_train_std,y_train)


"""
Out[3]: 
Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
   
   
"""


########################

"""
7) Lasso model results
Now that you've trained the Lasso model, you'll score its predictive capacity (R2) 
on the test set and count how many features are ignored because their coefficient is reduced to zero.

The X_test and y_test datasets have been pre-loaded for you.

The Lasso() model and StandardScaler() have been instantiated as la and scaler respectively and 
both were fitted to the training data.
"""

# Scale the test set
X_test_std = scaler.transform(X_test)

# Calculate the coefficient of determination (R squared) on the scaled test set
r_squared = la.score(X_test_std, y_test)
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))

# Create a list that has True values when coefficients equal 0
zero_coef = la.coef_ == 0

# Take the sum of this list
n_ignored = np.sum(zero_coef)
print("The model has ignored {} out of {} features.".format(n_ignored, len(la.coef_)))


"""
OUTPUT:

The model can predict 84.7% of the variance in the test set.

The model has ignored 82 out of 91 features.


Good! We can predict almost 85% of the variance in the BMI value using just 9 out of 91 of the features.
The R^2 could be higher though.

"""


#########################

"""
8) Adjusting the regularization strength
Your current Lasso model has an R2 score of 84.7%. When a model applies overly powerful regularization 
it can suffer from high bias, hurting its predictive power.

Let's improve the balance between predictive power and model simplicity by tweaking the alpha parameter.

Find the highest value for alpha that keeps the R2 value above 98% from the options: 1, 0.5, 0.1, and 0.01.

"""
# Find the highest alpha value with R-squared above 98%
la = Lasso(alpha=0.1, random_state=0)

# Fits the model and calculates performance stats
la.fit(X_train_std, y_train)
r_squared = la.score(X_test_std, y_test)
n_ignored_features = sum(la.coef_ == 0)

# Print peformance stats
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))
print("{} out of {} features were ignored.".format(n_ignored_features, len(la.coef_)))

"""
OUTPUT:
    The model can predict 98.3% of the variance in the test set.
    64 out of 91 features were ignored.

Wow! We this more appropriate regularization strength we can predict 98% of the variance in the 
BMI value while ignoring 2/3 of the features.
"""