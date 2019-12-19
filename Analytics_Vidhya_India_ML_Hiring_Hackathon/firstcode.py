#  Import All Important Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import nan


#  Input Train Data Set
dataset_train=pd.read_csv("train2.csv")


#  Input Test Data Set
dataset_test=pd.read_csv("test2.csv")



                                        # #Feature Engineering Started
                                                                             
                                        
                                        
# creating new varialbe for m1 to m12   to m_all

dataset_train['m_all'] = dataset_train['m12'].replace([1,2,3,4,5,6,7,8,9,10,11,12],1)
dataset_test['m_all'] = dataset_test['m12'].replace([1,2,3,4,5,6,7,8,9,10,11,12],1)


dataset_train=dataset_train.drop(['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12',], axis=1) 
dataset_test=dataset_test.drop(['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12',], axis=1) 





#Creating monthly emi bins to get all the persons whose emi is very high
'''
bins=[0,600,10000,]
group=['Low','high']
dataset_train['unpaid_bal_bin_mon']=pd.cut((dataset_train['unpaid_principal_bal']/dataset_train['loan_term']),bins,labels=group)
Income_bin=pd.crosstab(dataset_train['unpaid_bal_bin_mon'],dataset_train['m13'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')

dataset_test['unpaid_bal_bin_mon']=pd.cut((dataset_test['unpaid_principal_bal']/dataset_test['loan_term']),bins,labels=group)

dataset_train['unpaid_bal_bin_mon'] = dataset_train['unpaid_bal_bin_mon'].replace(['Low'],0)
dataset_train['unpaid_bal_bin_mon'] = dataset_train['unpaid_bal_bin_mon'].replace(['high'],1)
dataset_test['unpaid_bal_bin_mon'] = dataset_test['unpaid_bal_bin_mon'].replace(['Low'],0)
dataset_test['unpaid_bal_bin_mon'] = dataset_test['unpaid_bal_bin_mon'].replace(['high'],1)


dataset_train=dataset_train.drop(['unpaid_principal_bal'], axis=1)
dataset_test=dataset_test.drop(['unpaid_principal_bal'], axis=1)
'''



#Creating interest rate bins


dataset_train['interest_rate'].describe()


bins=[0,4.0,6.8]
group=['low','high']
dataset_train['interest_rate_bins']=pd.cut(dataset_train['interest_rate'],bins,labels=group)
Income_bin=pd.crosstab(dataset_train['interest_rate_bins'],dataset_train['m13'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')


dataset_test['interest_rate_bins']=pd.cut(dataset_test['interest_rate'],bins,labels=group)

dataset_train['interest_rate_bins'] = dataset_train['interest_rate_bins'].replace(['low'],0)
dataset_train['interest_rate_bins'] = dataset_train['interest_rate_bins'].replace(['high'],1)
dataset_test['interest_rate_bins'] = dataset_test['interest_rate_bins'].replace(['low'],0)
dataset_test['interest_rate_bins'] = dataset_test['interest_rate_bins'].replace(['high'],1)


dataset_train=dataset_train.drop(['interest_rate'], axis=1)
dataset_test=dataset_test.drop(['interest_rate'], axis=1)



# Creating borrower and co-borrower in one column

dataset_train['borrow_coborrow']=(((dataset_train['borrower_credit_score'])<=765) & ((dataset_train['co-borrower_credit_score']) <=765))
dataset_train['borrow_coborrow'] = dataset_train['borrow_coborrow'].replace([True],0)
dataset_train['borrow_coborrow'] = dataset_train['borrow_coborrow'].replace([False],1)

dataset_test['borrow_coborrow']=(((dataset_test['borrower_credit_score'])<=765) & ((dataset_test['co-borrower_credit_score']) <=765))
dataset_test['borrow_coborrow'] = dataset_test['borrow_coborrow'].replace([True],0)
dataset_test['borrow_coborrow'] = dataset_test['borrow_coborrow'].replace([False],1)




dataset_train=dataset_train.drop(['borrower_credit_score'], axis=1)
dataset_train=dataset_train.drop(['co-borrower_credit_score'], axis=1)
dataset_test=dataset_test.drop(['borrower_credit_score'], axis=1)
dataset_test=dataset_test.drop(['co-borrower_credit_score'], axis=1)



# Creating Loan values bins
'''
dataset_train['loan_to_value'].describe()

bins=[0,57,72,79,100]
group=['very_low','low','high','very_high']
dataset_train['loan_to_value_bin']=pd.cut(dataset_train['loan_to_value'],bins,labels=group)
Income_bin=pd.crosstab(dataset_train['loan_to_value_bin'],dataset_train['m13'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')

dataset_test['loan_to_value_bin']=pd.cut(dataset_test['loan_to_value'],bins,labels=group)


dataset_train['loan_to_value_bin'] = dataset_train['loan_to_value_bin'].replace(['very_low'],0)
dataset_train['loan_to_value_bin'] = dataset_train['loan_to_value_bin'].replace(['low'],1)
dataset_train['loan_to_value_bin'] = dataset_train['loan_to_value_bin'].replace(['high'],2)
dataset_train['loan_to_value_bin'] = dataset_train['loan_to_value_bin'].replace(['very_high'],3)

dataset_test['loan_to_value_bin'] = dataset_test['loan_to_value_bin'].replace(['very_low'],0)
dataset_test['loan_to_value_bin'] = dataset_test['loan_to_value_bin'].replace(['low'],1)
dataset_test['loan_to_value_bin'] = dataset_test['loan_to_value_bin'].replace(['high'],2)
dataset_test['loan_to_value_bin'] = dataset_test['loan_to_value_bin'].replace(['very_high'],3)



dataset_train=dataset_train.drop(['loan_to_value'], axis=1)
dataset_test=dataset_test.drop(['loan_to_value'], axis=1)

'''


                                        #Feature Engineering End
                                        
                                        
                                        

#dropping columns
dataset_train=dataset_train.drop(['loan_id','origination_date', 'first_payment_date'], axis=1) 
dataset_test=dataset_test.drop(['loan_id','origination_date', 'first_payment_date'], axis=1) 



# Creating Label Encoder ( genders male ,female ⇒ 0,1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset_train['source'] = le.fit_transform(dataset_train['source'])
dataset_train['financial_institution'] = le.fit_transform(dataset_train['financial_institution'])
dataset_train['loan_purpose'] = le.fit_transform(dataset_train['loan_purpose'])

dataset_test['source'] = le.fit_transform(dataset_test['source'])
dataset_test['financial_institution'] = le.fit_transform(dataset_test['financial_institution'])
dataset_test['loan_purpose'] = le.fit_transform(dataset_test['loan_purpose'])




# Create dummy variables for the columns for single columns and append it to the data
dataset_train = pd.concat([dataset_train, pd.get_dummies(dataset_train['source'], prefix='source_')], axis=1)
dataset_test = pd.concat([dataset_test, pd.get_dummies(dataset_test['source'], prefix='source_')], axis=1)
dataset_train = pd.concat([dataset_train, pd.get_dummies(dataset_train['financial_institution'], prefix='fin_inst')], axis=1)
dataset_test = pd.concat([dataset_test, pd.get_dummies(dataset_test['financial_institution'], prefix='fin_inst')], axis=1)
dataset_train = pd.concat([dataset_train, pd.get_dummies(dataset_train['loan_purpose'], prefix='loan_purpose')], axis=1)
dataset_test = pd.concat([dataset_test, pd.get_dummies(dataset_test['loan_purpose'], prefix='loan_purpose')], axis=1)

#dropping columns
dataset_train=dataset_train.drop(['source','financial_institution', 'loan_purpose'], axis=1) 
dataset_test=dataset_test.drop(['source','financial_institution', 'loan_purpose'], axis=1) 

# Preparing x and y 
y=dataset_train.m13  
dataset_train=dataset_train.drop(['m13'],axis=1)
x=dataset_train

'''
# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
x=sc_X.fit_transform(x)
dataset_test=sc_X.transform(dataset_test)

'''

#splitting the data into Training and Testing Data
from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv=train_test_split(x,y,test_size=0.3,random_state=0)


#Machine Learning Model (K Fold CV)

                      
                        
                        
cross_vald=10



# 7. XGBoost with K fold validation

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
kf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
accuracy_xgb_kfold=0
for train_index,test_index in kf.split(x,y):

    # Fitting the data into model
    
    xtr,xvl = x.loc[train_index],x.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    classifier_xgb_kfold = XGBClassifier(n_jobs = -1)
    classifier_xgb_kfold.fit(xtr, ytr)
    
    # Predicting the test set result for 

    y_pref_xgb_kfold = classifier_xgb_kfold.predict(xvl)
    
    # Summing all the accuracies 
    
    accuracy_xgb_kfold += f1_score(yvl,y_pref_xgb_kfold)
    
# Taking the average of K fold    
    
accuracy_xgb_kfold/=cross_vald




from xgboost import plot_importance
# plot feature importance
plot_importance(classifier_xgb_kfold)
plt.show()

classifier_xgb_kfold.get_booster().get_score(importance_type="gain")






#Machine Learning Model (GRID SEARCH)


cross_vald = 10
scoring_mec = 'f1'


# 7. Grid Search with XGBoost

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

parameters = {
'base_score' : [0.5],# defrault = 0.5
'booster' : ['gbtree'], # {gbtree,gblinear,dart} 
'colsample_bylevel' : [1],# default = 1, rane(0.00001,1)
'colsample_bytree' : [1],# default = 1, rane(0.00001,1)
'gamma' : [0],# default = 0 range (0,infinity)
'learning_rate' : [0.1], # default =.3, range(0,1)
'max_depth' : [3], # default = 6, range(0,infinity),Increasing this value will make the model more complex and more likely to overfit.
'max_delta_step' : [0], # default = 0,range(0,infinity)
'min_child_weight' : [1],# default = 1,range(0,infinity)
'n_estimators' : [10],# int , default = 10, The number of trees in the forest.
'nthread'  : [None],# same
'objective' : ['binary:logistic'],# very important parameter check documentation
'reg_alpha'  : [0], #default = 0
'reg_lambda' : [1], # default = 1
'scale_pos_weight' : [1], # default = 1
'seed' : [None], #default = 0 , random number seed
'silent' : [True],# same
'subsample' : [1],# default = 1 ,range(0,1)

                                        }

classifier_xgb = XGBClassifier()
grid_search_xgb = GridSearchCV(estimator = classifier_xgb,
                           param_grid = parameters,
                           scoring = scoring_mec,
                           cv = cross_vald,
                           n_jobs = -1)
grid_search_xgb = grid_search_xgb.fit(x, y)
accuracy_xgb_grid = grid_search_xgb.best_score_
bestparam_xgb_grid = grid_search_xgb.best_params_


# 9. Grid Search with LIGHT_GadientBoostingMachines

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

parameters = {
'boosting_type':['gbdt'],# {'gbdt','dart','rf','goss'}
'class_weight':[None], # same
'colsample_bytree' : [.1],# float,default : 1
'importance_type':['split'], # same
'learning_rate':[0.1],# float ,default =.1
'max_depth':[1], # int ,default = -1
'min_child_samples' : [10], # int , default = 20
'min_child_weight' : [1.5], # float, default =.0001
'min_split_gain' : [.1],# float, default = 0
'n_estimators'  : [100], # int  , default = 100
'num_leaves' : [31],# int ,default= 31
'random_state' : [None],# same
'reg_alpha' : [0,.001],# float, default : 0
'reg_lambda' : [0],# float, default : 0
'silent' : [True],# same
'subsample' : [.001],# float, default =1
'subsample_for_bin' : [200000],# int ,default = 200000
'subsample_freq' : [0] # same      
        }

#parameter={}
classifier_lgbm = LGBMClassifier()
grid_search_lgbm = GridSearchCV(estimator = classifier_lgbm,
                           param_grid = parameters,
                           scoring = scoring_mec,
                           cv = cross_vald,
                           n_jobs = -1)
grid_search_lgbm = grid_search_lgbm.fit(x, y)
accuracy_lgbm_grid = grid_search_lgbm.best_score_
bestparam_lgbm_grid = grid_search_lgbm.best_params_




# 10. Grid Search with CatBoost

from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
'''
parameters = {
'boosting_type':['gbdt'],# {'gbdt','dart','rf','goss'}
'class_weight':[None], # same
'colsample_bytree' : [.1],# float,default : 1
'importance_type':['split'], # same
'learning_rate':[0.1],# float ,default =.1
'max_depth':[1], # int ,default = -1
'min_child_samples' : [10], # int , default = 20
'min_child_weight' : [1.5], # float, default =.0001
'min_split_gain' : [.1],# float, default = 0
'n_estimators'  : [100], # int  , default = 100
'num_leaves' : [31],# int ,default= 31
'random_state' : [None],# same
'reg_alpha' : [0,.001],# float, default : 0
'reg_lambda' : [0],# float, default : 0
'silent' : [True],# same
'subsample' : [.001],# float, default =1
'subsample_for_bin' : [200000],# int ,default = 200000
'subsample_freq' : [0] # same      
        }
'''
parameters={}
classifier_cat = CatBoostClassifier()
grid_search_cat = GridSearchCV(estimator = classifier_cat,
                           param_grid = parameters,
                           scoring = scoring_mec,
                           cv = cross_vald,
                           n_jobs = -1)
grid_search_cat = grid_search_cat.fit(x, y)
accuracy_cat_grid = grid_search_cat.best_score_
bestparam_cat_grid = grid_search_cat.best_params_


# Creating final submission file for general and knn

pred_test =grid_search_lgbm.predict(dataset_test)

submission=pd.read_csv("sample_submission2.csv")

submission['m13']=pred_test 

pd.DataFrame(submission, columns=['loan_id','m13']).to_csv('sub_file_gen.csv')

