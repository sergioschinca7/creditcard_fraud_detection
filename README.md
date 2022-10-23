# Credit Card Fraud Detection using Python

I use a dataset from kaggle, here is the link https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation.

# Code

Here, I use K-Nearest Neighbors, Kernel SVM (Support Vector Machine), Random Forest, XGBoost (after XGBoost you can find k-Fold Cross Validation code to apply it if you want to)
to classify the transaction into normal and abnormal types.

I got 99.95% accuracy in our credit card fraud detection. Finally, I found that K-Nearest Neighbors is the winner, getting f1-score = 0.8383
