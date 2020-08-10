"""
In this competition, you will leverage data and ML methods to improve market outcomes for insurance provider Zimnat,
by matching consumer needs with product offerings in the Zimbabwean insurance market.
Zimnat wants an ML model to use customer data to predict which kinds of insurance products to recommend to customers.
The company has provided data on nearly 40,000 customers who have purchased two or more insurance products from Zimnat.

Your challenge: for around 10,000 customers in the test set, you are given all but one of the products they own,
and are asked to make predictions around which products are most likely to be the missing product.

This same model can then be applied to any customer to identify insurance products that might be useful to them given
their current profile.


For both train and test, each row corresponds to a customer, assigned a unique customer ID (‘ID’).
There is some information on the customer (when they joined, birth year etc).
The customer’s occupation (‘occupation_code’) and occupation category (‘occupation_category_code’) are also provided,
along with the branch code of the office they visit. The final 21 columns correspond to the 21 products on offer.

In Train, there is a 1 in the relevant column for each product that a customer has. Test is similar, except that for
each customer ONE product has been removed (the 1 replaced with a 0). Your goal is to build a model to predict the
missing product.

SampleSubmission shows the required submission format. For each customer ID, for each product, you must predict the
likelihood that that product is one in use by the customer. Notice that the sample submission contains 1s for the
products that are included in the test set, so that you can focus on the unknown products

**Leave your predictions as probabilities with values between 0 and 1 and do not round them to 0s or 1s.
The error metric for this competition is the log loss.

**50% of the test set is hidden (with Zindi). Could have the same differences which is seen between test and train set
here.

Issues so far:
1) Submission format/what data to include where
2) Wrong formatting of data (Gender)
3) Imbalanced datset
4) Recommender system vs logistic regression (multi-class v.s. multi-label?)
5) Job in test but not in training set
6) True vs untrue values
"""
"""
Analysis Log
10/08/2020: Load, cleaning, EDA.

"""

import os
import pandas as pd
import numpy as np
import datetime


# Loading data.
wd = os.getcwd()
data_dir = os.path.join(wd, 'data')

train = pd.read_csv(data_dir + '/train.csv',
                    parse_dates=['join_date'],
                    dayfirst=True)
test = pd.read_csv(data_dir + '/test.csv',
                   parse_dates=['join_date'],
                   dayfirst = True)
n_sub = pd.read_csv(data_dir + '/samplesub.csv')

# Changing datatypes to categorical data
for var in train.columns.drop('join_date'): # Checked to have the same columns.
    train[var] = train[var].astype('category')
    test[var] = test[var].astype('category')

# Extracting test set IDs so that can separate the training set and test set when needed after data cleaning
test_ids = test.ID

# Preliminary EDA
train_desc = train.describe()
test_desc = test.describe()
cols = train.columns
prod_code = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR',
             'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
             'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D',
             'J9JW', 'GHYX', 'ECY3']

# Some notes: Missing values, earliest date wrong. Date data could be dirty, need to check for inconsistencies [Done]
# Training set: Has 3 rows whose join date is in august [Done]
# Same top branches in both train/test set
# Many job occupations, 6 categories of jobs. Different job occupations in train and test set.
# Preference for products in both test and trainset seem to be very similar. Most popular products seem to be: K6QO,
# QBOL and RIBP.


# Data cleaning
train.isnull().sum()
train = train.dropna()
test.isnull().sum()
test = test.dropna()

train = train[~(train.join_date > pd.to_datetime('2020-07-01'))] # Removing dates beyond 1st July 2020

train.marital_status.unique() # f category. What is it?
test.marital_status.unique()
train.marital_status.replace('f', "F", inplace=True) # Changing f to F because that's the format in test set.

set1 = set(train.branch_code.unique()) # Checking differences in branch_codes between test/train sets.
set2 = set(test.branch_code.unique())
np.setdiff1d(set2, set1) # Checked both ways

set1 = set(train.occupation_code.unique()) # Checking differences in branch_codes between test/train sets.
set2 = set(test.occupation_code.unique())
np.setdiff1d(set1, set2) # Checked both ways. Some in train but not test, vice versa

train.groupby('')

# Basic feature engineering on both train and test sets.
# Join date --> Year, Month, Date, Quarter
# Age of join vs age as of 2020


# Find a way to suit training set to submission format. Do we predict for all values, then re-normalise based on the
# products which were not purchased yet? Removal of a random item --> pipe into new col? What should that new columns
# state? Need to look into whether the probabilities would still make sense or not + depend on public leaderboard.


# 