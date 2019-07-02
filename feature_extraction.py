"""
Manual feature extraction I
You want to compare prices for specific products between stores. The features in the pre-loaded dataset sales_df are:
storeID, product, quantity and revenue. The quantity and revenue features tell you how many items of a particular
product were sold in a store and what the total revenue was. For the purpose of your analysis it's more interesting
to know the average price per product.

"""

# Calculate the price from the quantity sold and revenue
sales_df['price'] = sales_df['revenue']/sales_df['quantity']

# Drop the quantity and revenue features
reduced_df = sales_df.drop(['quantity','revenue'], axis=1)

print(reduced_df.head())

"""
Good job! When you understand the dataset well, always check if you can calculate relevant features and drop irrelevant ones.

      storeID  product     price
    0       A   Apples  5.135616
    1       A  Bananas  3.365105
    2       A  Oranges  5.317020
    3       B   Apples  5.143417
    4       B  Bananas  3.898517

"""


"""
2. Manual feature extraction II
You're working on a variant of the ANSUR dataset, height_df, where a person's height was measured 3 times. 
Add a feature with the mean height to the dataset and then drop the 3 original features.


"""


# Calculate the mean height
height_df['height'] = height_df[['height_1','height_2','height_3']].mean(axis=1)

# Drop the 3 original height features
reduced_df = height_df.drop(['height_1','height_2','height_3'], axis=1)

print(reduced_df.head())


"""
   weight_kg    height
0       81.5  1.793333
1       72.6  1.696667
2       92.9  1.740000
3       79.4  1.670000
4       94.6  1.913333

Great! You've calculated a new feature that is still easy to understand compared to, for instance, principal components.

"""


"""
3. Calculating Principal Components

You'll visually inspect a 4 feature sample of the ANSUR dataset before and after PCA using Seaborn's pairplot().
This will allow you to inspect the pairwise correlations between the features.

The data has been pre-loaded for you as ansur_df.

"""

# Create a pairplot to inspect ansur_df
sns.pairplot(ansur_df)

plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create the scaler and standardize the data
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Create the PCA instance and fit and transform the data with pca
pca = PCA()
pc = pca.fit_transform(ansur_std)

# This changes the numpy array output back to a dataframe
pc_df = pd.DataFrame(pc, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])

# Create a pairplot of the principal component dataframe
sns.pairplot(pc_df)
plt.show()

"""
4) PCA on a larger dataset
You'll now apply PCA on a somewhat larger ANSUR datasample with 13 dimensions, once again pre-loaded as ansur_df. 
The fitted model will be used in the next exercise. Since we are not using the principal components 
themselves there is no need to transform the data, instead, it is sufficient to fit pca to the data.

"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale the data
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Apply PCA
pca = PCA()
pca.fit(ansur_std)

"""
Great! You've fitted PCA on our 13 feature datasample. Now let's see how the components explain the variance.


"""


"""
5) PCA explained variance
You'll be inspecting the variance explained by the different principal components of the pca instance 
you created in the previous exercise.

"""

# Inspect the explained variance ratio per component
print(pca.explained_variance_)

# Print the cumulative sum of the explained variance ratio
print(pca.explained_variance_ratio_.cumsum())

"""

Explained Variance: 

[0.61449404 0.19893965 0.06803095 0.03770499 0.03031502 0.0171759
 0.01072762 0.00656681 0.00634743 0.00436015 0.0026586  0.00202617
 0.00065268]

Cumulative Sum: 

[0.61449404 0.81343368 0.88146463 0.91916962 0.94948464 0.96666054
 0.97738816 0.98395496 0.99030239 0.99466254 0.99732115 0.99934732
 1.        ]
 
 
"""