import pandas as pd

"""
Basics:

shape (row,columns) 

"""

"""
Removing features without variance. 

"""

pokemon_df.describe()

"""
              HP     Attack     Defense  Generation
count  160.00000  160.00000  160.000000       160.0
mean    64.61250   74.98125   70.175000         1.0
std     27.92127   29.18009   28.883533         0.0
min     10.00000    5.00000    5.000000         1.0
25%     45.00000   52.00000   50.000000         1.0
50%     60.00000   71.00000   65.000000         1.0
75%     80.00000   95.00000   85.000000         1.0
max    250.00000  155.00000  180.000000         1.0


Here Generation column has no variance, all the data in the column holds the same value. 
"""

# Remove the feature without variance from this list
number_cols = ['HP', 'Attack', 'Defense']

# Leave this list as is
number_cols = ['HP', 'Attack', 'Defense']

# Remove the feature without variance from this list
non_number_cols = ['Name', 'Type', 'Legendary']

# Create a new dataframe by subselecting the chosen features
df_selected = pokemon_df[number_cols + non_number_cols]

# Prints the first 5 lines of the new dataframe
print(df_selected.head())

#Find the non-numeric feature without variance and remove its name from the list assigned to non_number_cols. Non - null columns

pokemon_df.describe(exclude="number")

"""
  Name   Type Legendary
count         160    160       160
unique        160     15         1
top     Electrode  Water     False
freq            1     31       160

Here Legendary has no variance, it has the same values. So we can remove the values.

"""


###########

"""
Feature Selection Vs Feature Extraction:

Visually detecting redundant features
Data visualization is a crucial step in any data exploration. Let's use Seaborn to explore some samples of the US Army
ANSUR body measurement dataset

"""

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_1, hue='Gender', diag_kind='hist')

# Show the plot
plt.show()



# Remove one of the redundant features
ansur_df_1.drop('stature_m', axis=1, inplace=True)

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_1, hue='Gender')

# Show the plot
plt.show()




# Remove the redundant feature
ansur_df_2.drop('n_legs',axis=1, inplace=True)

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_2, hue='Gender', diag_kind='hist')

# Show the plot
plt.show()


###########################

"""
t-SNE visualization of high-dimensional data

Fitting t-SNE to the ANSUR data
t-SNE is a great technique for visual exploration of high dimensional datasets. In this exercise, you'll apply it to 
the ANSUR dataset. You'll remove non-numeric columns from the pre-loaded dataset df and fit TSNE to his numeric dataset

"""


# Non-numerical columns in the dataset
non_numeric = ['Branch', 'Gender', 'Component']

# Drop the non-numerical columns from df
df_numeric = df.drop(non_numeric, axis=1)

# Create a t-SNE model with learning rate 50
m = TSNE(learning_rate=50)

# Fit and transform the t-SNE model on the numeric dataset
tsne_features = m.fit_transform(df_numeric)
print(tsne_features)


"""
OUTPUT:

[[-16.58397865  47.53944778]
 [  8.19966698  34.40295792]
 [ -1.89923191  26.82118797]
 ...
 [ -0.55705094   7.19647169]
 [ 26.71787643   3.33203149]
 [  1.1791991  -43.090065  ]]
 
t-SNE reduced the more than 90 features in the dataset to just 2 which we can now plot.

"""

