# Find a way to suit training set to submission format. Do we predict for all values, then re-normalise based on the
# products which were not purchased yet? Removal of a random item --> pipe into new col? What should that new columns
# state? Need to look into whether the probabilities would still make sense or not + depend on public leaderboard.


"""
Log Loss vs Accuracy
Accuracy is the count of predictions where your predicted value equals the actual value. Accuracy is not always a
good indicator because of its yes or no nature.

Log Loss takes into account the uncertainty of your prediction based on how much it varies from the actual label.
This gives us a more nuanced view into the performance of our model.

Logarithmic loss (related to cross-entropy) measures the performance of a classification model where the prediction
input is a probability value between 0 and 1. The goal of our machine learning models is to minimize this value. A
perfect model would have a log loss of 0. Log loss increases as the predicted probability diverges from the actual
label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high
log loss. There is a more detailed explanation of the justifications and math behind log loss here.

The graph below shows the range of possible log loss values given a true observation (isDog = 1). As the predicted
probability approaches 1, log loss slowly decreases. As the predicted probability decreases, however, the log loss
increases rapidly. Log loss penalizes both types of errors, but especially those predications that are confident
and wrong!
"""

"""
No cross-validation was conducted simply because one of the labels has one. This was oversight on my part, and I should
have ensured that there were at least a few per label to ensure that i could still have all classes in the validation 
sets. As such, for all my hyperparameter tuning, I chose settings which could lead me to more generalised model.
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
from sklearn.metrics import log_loss
from sklearn import tree
from sklearn.utils import class_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import graphviz
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import lightgbm as lgb

# Initialising values
SEED = 42

# Loading data.
wd = os.getcwd()
data_dir = os.path.join(wd, 'data')

train = pd.read_csv(data_dir + '/train_preproc.csv')
train_id = train.pop('ID')
test = pd.read_csv(data_dir + '/test_preproc.csv')
test_id = test.pop('ID')

# Checking out test set again in view of missing data. Fill in missing data with mean and mode.
test.isnull().any()
test['purchase_age'].fillna(test['purchase_age'].mean(),inplace=True)
test.loc[test['join_year'].isna(),'join_year'] = 2019.0 # Manually calculated mode of both join_year and join_month.
test.loc[test['join_month'].isna(),'join_month'] = 5.0
missing_row = test[test.isnull().any(axis=1)]


#Remember to convert the categorical variables to categories
datasets = [train, test]
col_names = train.columns
cat_var = col_names.drop(['birth_year', 'purchase_age', 'now_age', 'prod_bought']) # birth year change to numerical bc bin

for ds in datasets:
    if ds.shape[1] == 34:
        ds[cat_var.drop('prod_removed')] = ds[cat_var.drop('prod_removed')].astype('category')
        ds.info()
    else:
        ds[cat_var] = ds[cat_var].astype('category')
        ds.info()

# Checking out the clusters and the product codes
PROD_CODES = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR',
             'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
             'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D',
             'J9JW', 'GHYX', 'ECY3']
cluster_x = train.query('prod_cluster == 6')[PROD_CODES]

# Creating datasets without products
train_no_prod = train.drop(PROD_CODES, axis=1)
test_no_prod = test.drop(PROD_CODES, axis=1)

# Rearranging trainset
train_prod_removed = train.pop('prod_removed')
train.insert(34,'prod_removed', train_prod_removed)
train.isnull().any() # Last check for missing value before training

# Observing distribution of labels
train_label_dist = train['prod_removed'].value_counts(normalize=True).to_dict() # Convert to dict for weights


# Logistic regression. Change value of X_train to use train_no_prod instead of train to remove clustering
def prep_train_sets(df):
    X_train = df.drop('prod_removed', axis=1)
    X_train = pd.get_dummies(X_train, columns=['sex', 'marital_status', 'birth_year', 'branch_code',
                          'join_year','prod_cluster'],drop_first=True)
    Y_train = df['prod_removed'].copy()
    return X_train, Y_train



# Sizing up whether clusters will help or be detrimental to prediction. Try 3 variants of training dataset. Use best
# one to predict Y and submit preliminary score as baseline.
X_train, y_train = prep_train_sets(train) # This X_train and y_train to be used for rest of code.
log_reg_clf = LogisticRegression(penalty='l1', # want to use l1 so that everything else will shrink. See signifiacant values
                                 solver='saga',
                                 max_iter=1000,
                                 n_jobs=-1,
                                 multi_class='multinomial',
                                 verbose=2,
                                 random_state=SEED).fit(X_train,Y_train)

log_reg_clf.score(X_train, y_train) # With prod_codes = 0.822, Without prod_codes = 0.772, without prod_cluster = 0.821
log_loss(y_true=y_train, y_pred=log_reg_clf.predict_proba(X_train))
# With prod_codes = 0.737
# Without prod_codes = 0.919
# Without prod_cluster = 0.7398

# Identifying useful coefficients
def make_coef_df(fitted_model):
    coef_df = pd.DataFrame(fitted_model.coef_, index=fitted_model.classes_, columns=X_train.columns)
    return coef_df

coef_df = make_coef_df(log_reg_clf)
coef_df.to_csv('coefficients_logistic_regression.csv', index=True)

# Predicting probabilities using best set of data to come up with training set
test_pred = log_reg_clf.predict_proba(pd.get_dummies(test, columns=['sex', 'marital_status', 'birth_year', 'branch_code',
                      'occupation_category_code', 'join_year',
                      'join_month', 'join_quarter_CY', 'join_quarter_FY','prod_cluster'],drop_first=True))

pred_df = pd.DataFrame(test_pred,columns=log_reg_clf.classes_)
pred_df.shape


# Removing variables which doesn't seem to contribute much to prediction based on logistic regression.
# Removing useless variables found during ML
def remove_useless_variables(df, list_of_useless_var):
    for var in list_of_useless_var:
        df.pop(var)

remove_useless_variables(train,['join_month', 'join_quarter_FY', 'join_quarter_CY', 'occupation_category_code'])
remove_useless_variables(test,['join_month', 'join_quarter_FY', 'join_quarter_CY', 'occupation_category_code'])


# Random Forest initialisation. If want to try the other datasets, need to manually change the dataset used.
X_train, y_train = prep_train_sets(train)

class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train),
                                                  y=y_train)
class_weights = dict(zip(np.unique(y_train),class_weights))

rf_clf = RandomForestClassifier(criterion='entropy',
                                n_estimators= 500,
                                min_samples_split=3,
                                min_samples_leaf=3,
                                oob_score=True,
                                max_features=None,
                                n_jobs=-1,
                                verbose=2,
                                class_weight=class_weights,
                                random_state=SEED).fit(X_train, y_train)

rf_clf.oob_score_
# Vanilla: 0.854; 0.298??
# Increase Trees (1500): No change
# Min Samples Split and Leaf = 3: 0.8585/0.294
# Using max_features total columns/3: 0.8495/0.412
# Using max_features total columns/2: 0.852/0.394
# Using best combination as seen above: 0.863/0.2935
# Using correct class weights with best model: 0.819/0.292

log_loss(y_true=y_train, y_pred=rf_clf.predict_proba(X_train)) #Log loss of best model = 0.2935

rf_clf.score(X_train, y_train)

# Predicting probabilities using best set of data to come up with training set
test_pred = rf_clf.predict_proba(pd.get_dummies(test, columns=['sex', 'marital_status', 'birth_year', 'branch_code',
                                 'join_year','prod_cluster'],drop_first=True))

pred_df = pd.DataFrame(test_pred,columns=rf_clf.classes_)
pred_df.shape


# Plotting Tree.
#dot_data = export_graphviz(clf,
##                           out_file=None,
 #                          feature_names=X_train.columns,
#                           rounded=True,
#                           filled=True)
#graph = graphviz.Source(dot_data)
#graph
#graph.format = 'png'
#graph.render('dtree_render',view=True)

#graph = graphviz.Source(dot_data)
#png_bytes = graph.pipe(format='png')
#with open("".join(wd,'dtree_pipe.png'),'wb') as f:
#    f.write(png_bytes)


# Catboost
cat_features = train.columns.drop(['prod_cluster','purchase_age', 'now_age', 'prod_bought', 'prod_removed'])

cb_clf = CatBoostClassifier(iterations=1000,
                            depth=6,
                            loss_function='MultiClass',
                            random_seed=SEED,
                            verbose=True)

train['join_year'] = train['join_year'].astype('int') # Because of catboost requirements.
test['join_year'] = test['join_year'].astype('int')
cb_clf.fit(train.drop(['prod_removed','prod_cluster'],axis=1), train['prod_removed'], cat_features=cat_features,metric_period=5,verbose=True,early_stopping_rounds=100)


# make the prediction using the resulting model
test_pred = cb_clf.predict_proba(test.drop(['prod_cluster'],axis=1))

pred_df = pd.DataFrame(test_pred,columns=cb_clf.classes_)
pred_df.shape


