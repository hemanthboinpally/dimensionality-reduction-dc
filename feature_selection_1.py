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

###############

"""
5) Removing Features with Missing Values

.isna()
.sum()
/len(df)


"""

# Create a boolean mask on whether each feature less than 50% missing values.
mask = school_df.isna().sum() / len(school_df) < 0.5

# Create a reduced dataset by applying the mask
reduced_df = school_df.loc[:,mask]

print(school_df.shape)
print(reduced_df.shape)

"""
city            0
zipcode         0
csp_sch_id      0
sch_id          0
sch_name        0
sch_label       0
sch_type        0
shared        115
complex       129
label           0
tlt             0
pl              0



zipcode       0.000000
csp_sch_id    0.000000
sch_id        0.000000
sch_name      0.000000
sch_label     0.000000
sch_type      0.000000
shared        0.877863
complex       0.984733
label         0.000000
tlt           0.000000

(131, 21)
(131, 19)

"""



###########################


"""
6) Pairwise Correlation

Features that are related 
.corr()


"""
cmap = sns.diverging_palette(h_neg=10,
h_pos=240,
as_cmap=True)

# Create the correlation matrix
corr = ansur_df.corr()

# Draw the heatmap
sns.heatmap(corr,  cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()



#Mask the upper triangle.

# Create the correlation matrix
corr = ansur_df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Add the mask to the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()

"""
         Elbow rest height  Wrist circumference  Ankle circumference  Buttock height  Crotch height
Elbow rest height             1.000000             0.294753             0.301963       -0.007013      -0.026090
Wrist circumference           0.294753             1.000000             0.702178        0.576679       0.606582
Ankle circumference           0.301963             0.702178             1.000000        0.367548       0.386502
Buttock height               -0.007013             0.576679             0.367548        1.000000       0.929411
Crotch height                -0.026090             0.606582             0.386502        0.929411       1.000000

"""
###################

"""
Automated Removal of Highly Correlated Variables


"""

# Calculate the correlation matrix and take the absolute value
corr_matrix = ansur_df.corr().abs()

# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)

# List column names of highly correlated features (r > 0.95)
to_drop = [c for c in tri_df.columns if any(tri_df[c] >  0.95)]

# Drop the features in the to_drop list
reduced_df = ansur_df.drop(to_drop, axis=1)

print("The reduced dataframe has {} columns.".format(reduced_df.shape[1]))

####################


"""
You're right! While the example is silly, you'll be amazed how often people misunderstand correlation vs causation.

Example of why not correlated columns means the same, remove the column wisely.

"""