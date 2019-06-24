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