#############

"""
1) Testing you model for Overfitting

"""

# Import SVC from sklearn.svm and accuracy_score from sklearn.metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create an instance of the Support Vector Classification class
svc = SVC()

# Fit the model to the training data
svc.fit(X_train, y_train)

# Calculate accuracy scores on both train and test data
accuracy_train = accuracy_score(y_train, svc.predict(X_train))
accuracy_test = accuracy_score(y_test, svc.predict(X_test))

print("{0:.1%} accuracy on test set vs. {1:.1%} on training set".format(accuracy_test, accuracy_train))


"""
OUTPUT:
49.7% accuracy on test set vs. 100.0% on training set

Looks like the model is badly overfit on the training data. On unseen data it performs worse than a random selector 
would.

"""


###############

"""

2) Accuracy after dimensionality reduction - Can reduce the overfit after the dimensionality reduction

You'll reduce the overfit with the help of dimensionality reduction. In this case, you'll apply a rather drastic from 
of dimensionality reduction by only selecting a single column that has some good information to distinguish between genders. 
You'll repeat the train-test split, model fit and prediction steps to compare the accuracy on test vs. training data.

All relevant packages and y have been pre-loaded.



"""


# Assign just the 'neckcircumferencebase' column from ansur_df to X
X = ansur_df[['neckcircumferencebase']]

# Split the data, instantiate a classifier and fit the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
svc = SVC()
svc.fit(X_train, y_train)

# Calculate accuracy scores on both train and test data
accuracy_train = accuracy_score(y_train, svc.predict(X_train))
accuracy_test = accuracy_score(y_test, svc.predict(X_test))

print("{0:.1%} accuracy on test set vs. {1:.1%} on training set".format(accuracy_test, accuracy_train))


"""
   OUTPUT:
    93.3% accuracy on test set vs. 94.9% on training set

Wow, what just happened!? On the full dataset the model is rubbish but with a single feature we can make 
good predictions? This is an example of the curse of dimensionality! The model badly overfits 
when we feed it too many features. It overlooks that neck circumference by itself is pretty different for males 
and females.

"""

#####################


"""
3) Finding a good variance threshold

You'll be working on a slightly modified subsample of the ANSUR dataset with just head measurements 
pre-loaded as head_df.

boxplot, .boxplot()

normalize, df/ df.mean()

check variance of columns, .var()

remove the columns with the lowest variance 

"""

# Create the boxplot
head_df.boxplot()

plt.show()


# Normalize the data
normalized_df = head_df / head_df.mean()

normalized_df.boxplot()
plt.show()


# Normalize the data
normalized_df = head_df / head_df.mean()

# Print the variances of the normalized data
print(normalized_df.var())


"""

Here the threshold is 1.0e-03 to two lowest variance columns.

headbreadth          1.678952e-03
headcircumference    1.029623e-03
headlength           1.867872e-03
tragiontopofhead     2.639840e-03
n_hairs              1.002552e-08
measurement_error    3.231707e-27
"""


#######################

"""
4) Features with low variance
In the previous exercise you established that 0.001 is a good threshold to filter out low variance 
features in head_df after normalization. Now use the VarianceThreshold feature selector to remove these features.

"""
from sklearn.feature_selection import VarianceThreshold

# Create a VarianceThreshold feature selector
sel = VarianceThreshold(threshold=0.001)

# Fit the selector to normalized head_df
sel.fit(head_df / head_df.mean())

# Create a boolean mask
mask = sel.get_support()

# Apply the mask to create a reduced dataframe
reduced_df = head_df.loc[:, mask]

print("Dimensionality reduced from {} to {}.".format(head_df.shape[1], reduced_df.shape[1]))


"""
OUTPUT:

Dimensionality reduced from 6 to 4.



"""