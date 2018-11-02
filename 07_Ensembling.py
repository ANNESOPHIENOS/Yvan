
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import random
import os
import subprocess

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing, linear_model, metrics, grid_search, tree, ensemble
from sklearn.ensemble import RandomForestClassifier


import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

SEED = 32

np.random.seed(SEED)
random.seed(SEED)


# In[2]:


data_path = './'


# ## Excercise 7: Ensembling
# 
# We're now in to the final lab session of the Applied Machine Learning. Well done! If you have followed through all of the slides and completed each excercise, you are now one of dunnhumby's experts in machine learning.
# 
# So far, we have talked a lot about the nuts and bolts of modelling; validation, evaluation metrics, linear models etc. These are all fundamental components of machine learning; very few ML tasks will be completed without these.
# 
# Now, we are ready to start using "state of the art" techniques; ensemble methods. Ensembling is essentially combining the predictions of multiple models; for example, building 100 descision trees and averaging the predictions. Ensemble methods are almost always included in the winning solutions of Kaggle competitions. If you want pure predictive power, these are the methods for you.
# 
# Recall that in our last lab sessions, we were using logistic regressions and decision trees to predict whether a customer would buy pancake mixes in the following year, based on shopping behaviour of the previous year. Let's continue using this problem, but let's try fitting some ensemble methods to the data and see what happens to our test set performance...

# First, let's call the code that we ran last time to import and preprocess our data...

# In[3]:


# Like good programmers, we'll reuse our functions from the last lab session
def fill_missing(dataframe, value):
    """Fill missing values of a dataframe with value"""
    return dataframe.fillna(value)

def _determine_categoricals(dataset):
    """Return list of names for categorical columns in dataframe"""
    num_cols = train._get_numeric_data().columns
    return list(set(train.columns) - set(num_cols))
    
def create_dummy_variables(train, test):
    """Create dummmy variables for categorical columns in train and test (both must be dataframes)"""
    
    # Ensure that the columns are the same on both datasets
    assert(sum(train.columns == test.columns) == len(train.columns))
    
    # Temporarily combine the training and test sets
    full = pd.concat([train.reset_index(drop=True), test.reset_index(drop=True)], axis=0).reset_index(drop=True)
    
    # Determine the categorical columns
    cat_cols = _determine_categoricals(full)
    print('Dummies will be created for the following columns: {0:s}' .format(cat_cols))
    
    # Create dummy variables for those columns
    dummies_full = pd.get_dummies(full)

    # Return the full train and test sets with dummies inside
    train_wdummies = dummies_full.ix[0:train.shape[0]-1,:]
    test_wdummies = dummies_full.ix[train.shape[0]:dummies_full.shape[0],:]
    
    return train_wdummies, test_wdummies

def preprocess_data(train, test):
    """Preprocess a training and test set"""
    
    train = fill_missing(train, 0)
    test = fill_missing(test, 0)
    
    train, test = create_dummy_variables(train, test)
    
    return train, test

# We will just use data from this year (we're not trying to predict next year)
train = pd.read_csv(data_path + 'classification.csv') # This is the data from last month

# Separate out targets, features and ids
train_y = train['bought_pancakes'] # y variables
train_X = train.drop(['household', 'segmentation', 'bought_pancakes'], axis=1) # X features

# Create a training and two test sets
train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.7, random_state=SEED)
test_X, eval_X, test_y, eval_y = train_test_split(test_X, test_y, test_size=0.5, random_state=SEED)

# Preprocess the data
_, test_X = preprocess_data(train_X, test_X)
train_X, eval_X = preprocess_data(train_X, eval_X)


# Now we're ready to run a model. Recall that in the last lab session (6), we ran a DecisionTreeClassifier() over our data. With this model, we achieved an AUC of 0.66 and a log-loss of 0.69, which was similar to the regularised LogisticRegression() that we built in session 5.
# 
# What would happen if we built 50 DecisionTreeClassifiers and averaged the probabilities from each of them? This would be an example of __tree bagging__.
# 
# Thankfully, it's very easy to perform __bagging__ of any base learner thanks to sklearn's handy ```BaggingClassifier()``` function, which is part of the sklearn.ensemble library...

# In[12]:


n_estimators = 50

clf=RandomForestClassifier()

clf.fit(train_X, train_y)
preds = clf.predict_proba(test_X)[:, 1]
train_preds = clf.predict_proba(train_X)[:, 1]


# In[13]:


# Now let's evaluate our bagging model
auc = metrics.roc_auc_score(test_y, preds)
logloss = metrics.log_loss(test_y, preds)

train_auc = metrics.roc_auc_score(train_y, train_preds)
train_logloss = metrics.log_loss(train_y, train_preds)

print('Training Set Performance')
print('The Log Loss for this model on the training set is {0:.2f} ') .format(train_logloss)
print('The AUC for this model on the training set is {0:.2f} ') .format(train_auc)
print('')
print('Test Set Performance')
print('The Log Loss for this model on the test set is {0:.2f} ') .format(logloss)
print('The AUC for this model on the test set is {0:.2f} ') .format(auc)


# Ok nice! So by averaging the responses of 50 decision trees (rather than just building one) we've improved our test set log-loss from 0.69 to 0.57. That's a nice improvement. The AUC hasn't improved, but we haven't tuned any of the hyper-parameters yet.
# 
# At the moment, we are just using the default hyperparameters for both the DecisionTreeClassifier() and the BaggingClassifier(). 
# 
# In addition, we're actually not using the best implementation of tree bagging. Thankfully, sklearn has implemented ```RandomForestClassifier()```, which is the most commonly used tree bagging algorithm. Rather than specifying a base learner, Random Forest uses decision trees by default. 
# 
# Let's have a go at grid searching over some of the most important hyper-parameters of a Random Forest...

# I'll define quite a small grid search because I dont want it to take forever. Normally I would define a much larger range and have far more trees (n_estimators=1000) and then let it run overnight. But we don't have overnight so ...

# In[14]:


parameters = {
    'n_estimators' : [300],
    'max_depth' : [6,8],
    'max_features' : [3,4]
}

mod = ensemble.RandomForestClassifier(random_state=SEED)
clf = grid_search.GridSearchCV(mod, parameters, cv=2, n_jobs=1, verbose=3, scoring='roc_auc')
clf.fit(train_X, train_y)

# Now we can make predictions with the best model from the grid search
rf_preds = clf.predict_proba(test_X)[:, 1]
train_preds = clf.predict_proba(train_X)[:, 1]


# In[16]:


# Now let's evaluate our bagging model
auc = metrics.roc_auc_score(test_y, rf_preds)
logloss = metrics.log_loss(test_y, rf_preds)


train_auc = metrics.roc_auc_score(train_y, train_preds)
train_logloss = metrics.log_loss(train_y, train_preds)


print('Training Set Performance')
print('The Log Loss for this model on the training set is {0:.2f} ') .format(train_logloss)
print('The AUC for this model on the training set is {0:.2f} ') .format(train_auc)
print('')
print('Test Set Performance')
print('The Log Loss for this model on the test set is {0:.2f} ') .format(logloss)
print('The AUC for this model on the test set is {0:.2f} ') .format(auc)
print('The best parameters found by the grid search are {0:s} ') .format(clf.best_params_)


# Great! So a RandomForestClassifier() has given us our best test set performance to date. Our previous best log-loss was 0.41 and the Random Forest is now slightly better. However, the AUC is now 0.03 points better than our previous best model (the Logistic Regression). Good work!

# ##### Q1: Try increasing and descreasing the value of n_estimators. Firstly, what does n_estimators do? Secondly, what happens to the 1) time taken to fit 2) performance of the model when you increase/decrease the size of n_estimators? 

# ##### Q2: Take a look at the documentation for the RandomForestClassifier (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). Choose a few hyper-parameters that we haven't yet tuned and add them to the grid-search (you may wish to not grid-search over hyperparameters we have already tuned, to save time). Has your model improved after tuning these values?

# ## Excercise 7.2: Gradient Boosting Machines
# 
# So far in this lab, we have shown you two different ways to perform tree bagging. In addition to tree-bagging, there is another major machine learning algorithm that needs to be included in your inventory; Gradient Boosting.
# 
# Gradient Boosting is an additive algorithm that aims to improve on the performance of previously fit decision trees by focusing on their errors. By constantly focusing on the residuals of previous decision trees, gradient boosting machines get better over time.
# 
# Sklearn has implementations of gradient boosting in the way of the ```GradientBoostingClassifier()``` and the ```GradientBoostingRegressor()```. However, I'm not going to waste your time by taking you through that class, because it is not the best. 
# 
# By far, the best implementation of gradient boosting in Python (also available in R) is __xgboost__. Whilst it's not part of the sklearn library, it looks and feels a lot like sklearn. It's also now fully compatible - thanks to the lovely open source community - with all of the handy sklearn classes and functions.
# 
# Let's have a go at fitting an xgboost gradient boosting machine to our data...

# In[17]:


import xgboost


# In[ ]:


parameters = {
    'n_estimators' : [400],
    'max_depth' : [3,5,7],
    'subsample' : [0.5, 1.],
    'learning_rate' : [0.01]
}

mod = xgboost.XGBClassifier(seed=SEED)
clf = grid_search.GridSearchCV(mod, parameters, cv=2, n_jobs=1, verbose=3, scoring='roc_auc')
clf.fit(train_X, train_y)

# Now we can make predictions with the best model from the grid search
xgb_preds = clf.predict_proba(test_X)[:, 1]


# In[10]:


# Now let's evaluate our bagging model
auc = metrics.roc_auc_score(test_y, xgb_preds)
logloss = metrics.log_loss(test_y, xgb_preds)

print('Test Set Performance')
print('The Log Loss for this model on the test set is {0:.2f} ') .format(logloss)
print('The AUC for this model on the test set is {0:.2f} ') .format(auc)
print('The best parameters found by the grid search are {0:s} ') .format(clf.best_params_)


# OK! Without much tuning, we've got to an AUC of 0.69; the same as our previous best model, the Random Forest.
# 
# With a bit more time, I imagine that we could push the model to an AUC of 0.70. This is 4% AUC points better than the Logistic Regression. This may not seem like much, but it can mean a considerable amount for a client. For example, a 4% point improvement in retention rates can improve a large retailer's bottom line by millions of pounds.

# #### Saving our model and predictions
# 
# Thinking back to our original brief in the Logistic Regression lab, we were asked to predict where customers were going to buy pancake mixes next year. We used information about customers' purchasing behaviour from the previous years to build our predictive models. Now it's time to score up our test dataset!
# 
# We've run a couple of models now; a Logistic Regression, a Decision Tree, a Random Forest and a Gradient Boosting Machine (GBM). After evaluating and tuning each of these models, the Random Forest and the GBM has generated the best predictive performance on the test set.
# 
# Now, we need to make sure that we can deliver these predictions to our client. They have asked - for each Household ID - for a predicted probability and a raw predicted value (i.e. 1 if we predict they will buy and 0 if we predict that they won't).
# 
# We'll start by re-fitting the final model on the __whole__ dataset, so that we're using as much data as possible...

# In[11]:


# Then let's fit the model on the training and first test set
final_X = pd.DataFrame(np.vstack([train_X, test_X]), columns=train_X.columns)
final_y = np.concatenate([train_y, test_y])

mod = xgboost.XGBClassifier(seed=SEED)
# Set the parameters of xgboost to the best ones we found in the grid search
mod.set_params(**clf.best_params_)
mod.fit(final_X, final_y)


# We then need to check that our final model performs as well on the final evaluation set, to ensure that nothing major has gone wrong...

# In[12]:


eval_preds = mod.predict_proba(eval_X)[:,1]
auc = metrics.roc_auc_score(eval_y, eval_preds)
logloss = metrics.log_loss(eval_y, eval_preds)

print('Final Evaluation Set Performance')
print('The Log Loss for this model on the test set is {0:.2f} ') .format(logloss)
print('The AUC for this model on the test set is {0:.2f} ') .format(auc)


# That looks reasonable; it's the same as our test set performance! Now, we need to read in our scoring dataset. 
# 
# If you remember the original brief, this contains behavioural features for households between Jan 2015 to 2016. We are going to be predicting whether customers will purchase pancake mixes in February 2016, which hasn't happened yet ...

# In[13]:


# Load in the test data
final_eval = pd.read_csv(data_path + 'classification_test.csv') 

# Separate out targets, features and ids
final_eval_idx = final_eval['household']
final_eval_X = final_eval.drop(['household'], axis=1)

# Preprocess the data
_, final_eval_X = preprocess_data(train.drop(['household', 'segmentation', 'bought_pancakes'], axis=1), final_eval_X)

# Check that the columns are the same in both datasets
assert(sum(train_X.columns == final_eval_X.columns) == len(train_X.columns))


# Now we're ready to score up this year's set of customers for our client's mailer...

# In[14]:


final_probs = mod.predict_proba(final_eval_X)[:, 1]
final_preds = mod.predict(final_eval_X)


# Let's combine those predictions with our household_ids. Then we'll have something that can be delivered to our client...

# In[15]:


final_pred_df = pd.DataFrame(np.column_stack([final_probs, final_preds]), columns=['p(Buy Pancakes in Feb 16)', 'Will they Purchase?'])
final_pred_df = pd.concat([final_eval_idx, final_pred_df], axis=1)


# In[16]:


final_pred_df.head()


# Amazing! So - as requested by our client - we have now predicted the probability that each 2015 customer will buy pancakes in February 2016, using shopping data from January 2015 - 2016. We have (for each household) created a probability and a raw predicted value.
# 
# We're now ready to send this to the client. So we'll export that to a CSV. In addition, we'll create a pickle (i.e. binary on disk) of our final model, just in case we need to use it again at a later date.

# In[17]:


from sklearn.externals import joblib
final_pred_df.to_csv('./PancakeDay16_Propensity.csv', index=False) # Export the CSV to the current working directory
joblib.dump(mod, './PancakeDay16_XGBoost.pkl') # Export the XGBOOST to the current working directory


# We can now securely transfer the CSV to the people responsible for sending Pancake Day mailers to customers. We'll keep the model pickle to ourselves, as we need to be careful about sharing IP/coefficients.

# #### A note on regression problems
# 
# So far, we haven't discussed how to do regression problems (i.e. continuous targets) using trees, random forest and GBMs. 
# 
# Regression is just as easy to do as classification in sklearn. The code is *exactly* the same, except you will use a different model class, a different evaluation metric and you will use the ```.predict()``` method rather than the ```.predict_proba()``` method.
# 
# In particular, you can use the following model classes for regression problems:
# 
# 1. ```tree.DecisionTreeRegressor()```
# 1. ```ensemble.RandomForestRegressor()```
# 1. ```xgboost.XGBRegressor()```
# 
# Note that most of the hyper-parameters will be exactly the same as the ones we have already discussed.
# 
# It really is that easy.

# ## Conclusion
# 
# In this training, we've explained how you can guide algorithms through your data so that they learn to predict the future quickly and accurately.
# 
# We've seen the importance of defining your evaluation metric, validating your models on out-of-sample data and preprocessing your data appropriately.
# 
# We've also shown that - if you want a good model - you have to select features down. Whilst we tried many feature selection methods, we found that L1 Regularised Regression was an excellent method for identifying the powerful features amongst the noisy ones. 
# 
# As for models, we've seen that linear models can be both insightful and powerful, if they are built the correct way. Moreover, we've shown that decision trees can also be insightful and powerful, but prone to overfitting. We can overcome issues associated with decision trees if we grid-search over our hyper-parameters properly or combine them (using ensembling) in some way. However - whilst ensemble methods might be the best-in-class for prediction - they are less interpretable than linear models and (some) decision trees.
# 
# Each step that we take during the machine learning pipeline needs to be taken with care. Often, you should validate decisions by testing them in a model and seeing whether cross-validation performance increases or decreases; this is the nature of science.
# 
# Hopefully you have seen that - using sklearn and tools built by the open source community - you can achieve excellent, predictive and re-usable models in a relatively small amount of time. Now that you too are an expert in machine learning, I encourage you to join the open source community and share your knowledge as widely as possible!
