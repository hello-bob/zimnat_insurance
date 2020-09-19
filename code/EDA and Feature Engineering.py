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
16/08/2020: More EDA and feature engineering
23/08/2020: Feature engineering + clustering

"""

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from joblib import Parallel, delayed
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# Initialising values
SEED = 42

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
train_desc = train.describe(datetime_is_numeric=True)
test_desc = test.describe(datetime_is_numeric=True)
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


# Data cleaning. If want to exclude outliers, need to come back here to do the whole thing again.
train.isnull().sum()
train = train.dropna()
test.isnull().sum()

train = train[~(train.join_date > pd.to_datetime('2020-07-01'))] # Removing dates beyond 1st July 2020


train.marital_status.unique() # f category. What is it?
test.marital_status.unique() # Changing f to F because that's the format in test set.
train.marital_status.cat.rename_categories({'M':'M',
                                            'U':'U',
                                            'S':'S',
                                            'W':'W',
                                            'D':'D',
                                            'P':'P',
                                            'R':'R',
                                            'f':'F'},inplace=True)


set1 = set(train.branch_code.unique()) # Checking differences in branch_codes between test/train sets.
set2 = set(test.branch_code.unique())
np.setdiff1d(set2, set1) # Checked both ways

set1 = set(train.occupation_code.unique()) # Checking differences in branch_codes between test/train sets.
set2 = set(test.occupation_code.unique())
np.setdiff1d(set1, set2) # Checked both ways. Some in train but not test, vice versa

# How many products bought per person in each train/test set.
prod_per_pax_train = sum(train[prod_code].sum(axis=0))/len(train) # 2.277
prod_per_pax_test = sum(test[prod_code].sum(axis=0))/len(test) # 1.285.

# Frequencies of each product for training and test set
train_prod_freq = (train[prod_code].sum(axis=0)/len(train))*100
test_prod_freq = (test[prod_code].sum(axis=0)/len(test))*100
prod_freq_compare = pd.concat([train_prod_freq, test_prod_freq],
                              axis=1)
prod_freq_compare.columns = ['train', 'test'] # Quite varied frequency per product.
prod_freq_compare.sort_values('train',
                              ascending=False,
                              inplace=True) # Top 3 products are RVSZ, K6QO, QOBL. Next 2 are PYUQ and RIBP
prod_freq_compare_normed = prod_freq_compare.copy()
prod_freq_compare_normed['train'] = (prod_freq_compare_normed['train']/prod_freq_compare_normed['train'].sum())
prod_freq_compare_normed['test'] = (prod_freq_compare_normed['test']/prod_freq_compare_normed['test'].sum())
# After normalising each product to the number of products in each of the train and test set, the frequency of each
# product looks quite similar. Maximum difference is just 1.5% for RVSZ. I.e. per person, per product

# Maximum and minimum no of products purchased
print("Minimum number of products (Train): ", train[prod_code].sum(axis=1).min(),
      "Maximum number of products (Train): ", train[prod_code].sum(axis=1).max())

print("Minimum number of products (Test): ", test[prod_code].sum(axis=1).min(),
      "Maximum number of products (Test): ", test[prod_code].sum(axis=1).max())
# If want to go by the clustering method, notice that there are extreme numbers in training set i.e. some people buying
# 9 and 14 products. They could serve as outliers, hence might want to delete them from training set.

# Plotting it out
#plt.bar(train[prod_code].sum(axis=1).value_counts().index,
#        train[prod_code].sum(axis=1).value_counts().values,
#        label='train')
#plt.bar(test[prod_code].sum(axis=1).value_counts().index,
#        test[prod_code].sum(axis=1).value_counts().values,
#        label='test')
#plt.legend()
#plt.show()

# Some preliminary plots
#train[['ID','marital_status']].groupby('marital_status').count().plot(kind='bar') # Some marital status non-existent.
#train[['ID','sex']].groupby('sex').count().plot(kind='bar') # More males than females
#train[['ID','birth_year']].groupby('birth_year').count().plot(kind='bar') # Quite a normal distribution. 1985 peak
#train[['ID','branch_code']].groupby('branch_code').count().plot(kind='bar')
# Some branches more popular than the others. Some branches are almost non-existent
# Some occupation codes are grossly higher than others. e.g. T4MS. Some categories are non-existent.

# Combinations of products
def prod_comb(df):
    sorted_codes = sorted(prod_code) # So that all the codes will concatenate in order
    temp_df = df[sorted_codes].copy()
    for prod in temp_df.columns:
        temp_df[prod] = np.where(temp_df[prod] == 1, prod, '')
    comb = temp_df.sum(axis=1)
    return comb

train_comb = prod_comb(train)
train_comb_counts = train_comb.value_counts()
train_comb_counts.rename('train', inplace=True)

test_comb = prod_comb(test)
test_comb_counts = test_comb.value_counts()
test_comb_counts.rename('test', inplace=True)

comb_counts = pd.merge(train_comb_counts,
                       test_comb_counts,
                       how='outer',
                       left_on=train_comb_counts.index,
                       right_on=test_comb_counts.index)

# From this, can see that most combinations are in train which are not in test. This is in line with the fact that
# minimum products purchased by each person is 2. Need to take note that I will be predicting based on this as well.
# Perhaps I will need to observe the distribution again when I remove one product from each person in the training set.

# Some patterns of combinations are starting to come out already from training set

comb_counts.rename({'key_0':'prod_comb'},axis=1,inplace=True)



# Removing one product form test set and log it down. Remember that this is random, so need to set seed. Might need to
# train the whole model upon new ways of removing product in view of case where there's a common first product purchased
# etc etc. Possible to generate ensemble based on the randomness of this choice?
prod_only = train[prod_code].copy()
prod_only['prod_removed'] = np.empty((prod_only.shape[0],1), str)

# Something is wrong with the indexing. Look into it
def remove_one_prod(df):
    np.random.seed(SEED)
    temp_df = pd.DataFrame()
    for idx, row in df.iterrows():
        print(idx)
        mask = (row == 1)
        all_prod = row[mask].index # Get all products which a customer purchased)

        # Determining which product to remove and removing product.
        prod_to_remove = np.random.choice(a=all_prod, size=1).item()  # Select one product from list of all products.
        row[prod_to_remove] = 0 # Set that product to 0
        row['prod_removed'] = prod_to_remove # while storing removed product in prod_removed column

        # Append row to temp_df. Don't change values in place. Concatenate by series (columns), so transpose after
        temp_df = pd.concat([temp_df, row], axis=1, sort=False)

    temp_df = temp_df.T
    return(temp_df)




removed_prod = remove_one_prod(prod_only) # Execute function to remove one product from every individual
# Which is the product removed. Will be train_y. Add in ID so can leftjoin and prevent index mix-up
prod_removed = pd.concat([train['ID'], removed_prod['prod_removed']],axis=1)
train[prod_code] = removed_prod.drop('prod_removed', axis=1)

# Replacing the products in matrix with removed code
# From here onwards, the train and test set can be manipulated in the same way because the information is roughly the same
# i.e. one product removed.


# Basic feature engineering on both train and test sets. Combine train and test sets .
train_n_rows = train.shape[0] # Extract number of rows so that can use indexing to split the train and test sets again.
# As a reminder, to use this number, just comb_df.iloc[range(train_n_rows)]

train.info()

comb_df = pd.concat([train,test],axis=0,ignore_index=True)


comb_df['join_year'] = comb_df['join_date'].dt.year # Year that customer joined
comb_df['join_month'] = comb_df['join_date'].dt.month # Month that customer joined

# Calendar year quarter that customer joined
conditions = [comb_df['join_month']<4, ((comb_df['join_month']>=4) & (comb_df['join_month']<7)),
              ((comb_df['join_month']>=7) & (comb_df['join_month']<10))]
comb_df['join_quarter_CY'] = np.select(conditions, ['q1','q2','q3'], 'q4')

# Financial year quarter that customer joined
conditions = [comb_df['join_month']<4, ((comb_df['join_month']>=4) & (comb_df['join_month']<7)),
              ((comb_df['join_month']>=7) & (comb_df['join_month']<10))]
comb_df['join_quarter_FY'] = np.select(conditions, ['q4','q1','q2'], 'q3')

comb_df['purchase_age'] = comb_df['join_year']-comb_df['birth_year']
comb_df['now_age'] = 2020-comb_df['birth_year']
comb_df.pop('join_date') # Take note, if want to create more features based on join date, need to remove this line.
# plt.bar(comb_df['now_age'].value_counts().index, comb_df['now_age'].value_counts().values)

# Binning birth year, otherwise cannot RF will fail (too many categories)
year_bins = np.arange(1930,2005, 5)
year_bins = np.append(year_bins,2015) # Because max year in both sets is 2011. Max year in test set is 2001.
year_bins_labels = ["".join(['<',str(year)]) for year in year_bins]
comb_df.birth_year= pd.cut(comb_df.birth_year,bins=year_bins, labels=year_bins_labels[1:],right=False)


# Sum number of products bought (after removal of products for training set)
comb_df['prod_bought'] = comb_df[prod_code].sum(axis=1).astype('int64')

#Convert categories into categories
comb_df['join_month'] = comb_df['join_month'].astype('category')

# Dropping occupation code because there are so many levels. Might result in problems downstrean
comb_df.drop('occupation_code', axis=1, inplace=True)

# Convert all that is categorical to categorical variable before proceeding!
cat_var_convert = prod_code.copy()
cat_var_convert.extend(['birth_year', 'join_year', 'join_month', 'join_quarter_CY', 'join_quarter_FY'])

comb_df[cat_var_convert] = comb_df[cat_var_convert].astype('category')

comb_df.query('ID == "MOIL20O"')

#Split into train and test set again
train = comb_df.iloc[:train_n_rows].copy()
train= train.merge(prod_removed, how='left', on='ID') # This is the y variable to be predicted. Use multiclass methods.

# Remember that now for training set, will have 2 columns which should not be included in any training: ID and prod_removed
#train, valid = train_test_split(train, test_size=0.25,random_state=SEED)
test = comb_df.iloc[train_n_rows:].copy()


#Scale numerical variables for both train and test set. Only need to do for purchase_age, now_age, and prod_bought
train['purchase_age'] = preprocessing.scale(train['purchase_age'])
train['now_age'] = preprocessing.scale(train['now_age'])
train['prod_bought'] = preprocessing.scale(train['prod_bought'])

#valid['purchase_age'] = preprocessing.scale(valid['purchase_age'])
#valid['now_age'] = preprocessing.scale(valid['now_age'])
#valid['prod_bought'] = preprocessing.scale(valid['prod_bought'])

test['purchase_age'] = preprocessing.scale(test['purchase_age'])
test['now_age'] = preprocessing.scale(test['now_age'])
test['prod_bought'] = preprocessing.scale(test['prod_bought'])

# Clustering method for feature engineering. Cluster and compare cluster group distribution with comb_counts
# (for train only). Can tweak cluster numbers if needed. Rationale behind using clustering method for products only is
# because the formatting of the dataset will result in each row losing the information of the set of products which
# each individual bought.

# Finding best number of clusters. Best number of cluster was found to be:
train.columns
cost_list = []
cluster_list = [2,4,6,8,10,12,14,16] # 10 was first elbow. Just use 10 because any more will be v compute intensive. Time.
for clusters in cluster_list:
    km = KModes(n_clusters=clusters, init='Huang', n_init=5, verbose=2, random_state=SEED, n_jobs=-1)
    train_fit = km.fit(train[prod_code])
    cost = train_fit.cost_
    cost_list.append(cost)

plt.plot(cluster_list, cost_list)
plt.xlabel('number of cluster')
plt.ylabel('Cost')

# Code to compare clustering below
#train_comb_removed_prod = prod_comb(train[prod_code])
#compare_cluster = pd.concat([train_comb_removed_prod,cluster_series],axis=1)
#compare_cluster.columns = ['prod_combi', 'clusters']
#compare_cluster.clusters = compare_cluster.clusters.astype('category')
#cluster_1=compare_cluster.query('clusters == 3').sort_values('clusters')
#cc_groupby = compare_cluster.groupby(by='clusters',as_index=False, observed=True).count()

# Initialising model using best number of clusters using elbow method. Eventually drop cluster to 8 because of eird
# results despite drop in loss.
km = KModes(n_clusters=12, init='Huang', n_init=5, verbose=2, random_state=SEED, n_jobs=-1)
train_fit = km.fit(train[prod_code])
help(KModes)
# Apply clustering to all data sets.
train['prod_cluster'] = train_fit.predict(train[prod_code])
train['prod_cluster'] = train['prod_cluster'].astype('category')
train['prod_cluster'].value_counts()
#valid['prod_cluster'] = train_fit.predict(valid[prod_code])
#valid['prod_cluster'] = valid['prod_cluster'].astype('category')
#valid['prod_cluster'].value_counts()
test[prod_code].isnull().any()

test['prod_cluster'] = train_fit.predict(test[prod_code])
test['prod_cluster'] = test['prod_cluster'].astype('category')
test['prod_cluster'].value_counts()

# Saving as train test and validation set so no need to do the pre-proc again, unless I decide to change the clusters.
train.to_csv('train_preproc.csv', index=False)
test.to_csv('test_preproc.csv', index=False)
# test have to manually delete birth_year for GDR6OIV after export to excel because missing values.


# People clustering. Creates 6 types of people. Can consider if want to reduce dimensionality ttm
#cat_var = train.select_dtypes('category').columns
#cat_var_index = [train.columns.get_loc(c) for c in cat_var]

#kp = KPrototypes(n_clusters=6, init='Huang', n_init=5, verbose=2, random_state=SEED, n_jobs=-1)
#clusters = kp.fit_predict(train, categorical=cat_var_index) #
# train['ppl_cluster'] = pd.Series(clusters,name='ppl_cluster')

# Product + people clustering. KIV, might interfere with flexibility
#train.columns.drop('')
#cat_var.extend(prod_code)
#kp = KPrototypes(n_clusters=6, init='Huang', n_init=5, verbose=1, random_state=SEED, n_jobs=-1)
#clusters = kp.fit_predict(train.drop('ID', axis=1), categorical=cat_var)
#prod_ppl_cluster = pd.Series(clusters,name='prod_ppl_cluster')





# Apply SMOTENC on training set only IF GOT TIME. Otherwise, just proceed first. Anyways will need to come up with 2 sets,
# one with sampling, the other one without
#cat_index = [train.columns.get_loc(col) for col in train.columns if train[col].dtype != 'float64']

#sm = SMOTENC(categorical_features=cat_index, random_state=SEED)
#X_res, y_res = sm.fit_resample(X, y)