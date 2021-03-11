import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import gc

from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def detect_types(data):
    """
    Separates columns to numerical ones and categorical ones
    """
    numerical_preds=[]
    categorical_preds=[]
     
    
    for i in list(data):
        if(data[i].dtype=='object'):
            categorical_preds.append(i)
        else:
            numerical_preds.append(i)
    
    return numerical_preds,categorical_preds




def graph_exploration(feature_binned,target):
    """
    Function that visualises relationship between given binned variable and 
    binary target

    """
  
    if(sum(feature_binned.isnull())>0):
        feature_binned=feature_binned.cat.add_categories('NA').fillna('NA')
    
    result = pd.concat([feature_binned, target], axis=1)
    
    gb=result.groupby(feature_binned)
    counts = gb.size().to_frame(name='counts')
    final=counts.join(gb.agg({result.columns[1]: 'mean'}).rename(columns={result.columns[1]: 'target_mean'})).reset_index()
    final['pom.sanci']=np.log2((final['counts']*final['target_mean']+100*np.mean(target))/((100+final['counts'])*np.mean(target)))
        
    sns.set(rc={'figure.figsize':(15,10)})
    fig, ax =plt.subplots(2,1)
    sns.countplot(x=feature_binned, hue=target, data=result,ax=ax[0])
    sns.barplot(x=final.columns[0],y='pom.sanci',data=final,color="green",ax=ax[1])
    plt.show()
    
    
def graph_exploration_continuous(feature_binned,target):
    """
    Function that visualises relationship between given binned variable and 
    continuous target
    """


    if(sum(feature_binned.isnull())>0):
        feature_binned=feature_binned.cat.add_categories('NA').fillna('NA')
    
    plt.figure(figsize=(12,5))
    sns.boxplot(x=feature_binned,y=target,showfliers=False)
    plt.xticks(rotation='vertical')
    plt.show()
    

def plot_imp(dataframe,imp_type,ret=False,n_predictors=100):
    """
    Plots variables importance of features of lgbm model
    """
    plt.figure(figsize=(20,n_predictors/2))
    sns.barplot(x=imp_type, y="Feature", data=dataframe.sort_values(by=imp_type, ascending=False).head(n_predictors))
    plt.show()
    if ret==True:
        return list(dataframe.sort_values(by=imp_type, ascending=False).head(n_predictors)['Feature'])


def plot_roc(fpr,tpr,gini,label):

    plt.figure(figsize=(10,5))
    #plt.plot(fpr, tpr, label='ROC curve of validation set (area = %0.2f)' % (roc_auc))
    if (label=='valid'):
        plt.plot(fpr, tpr, label='ROC curve of valid set (GINI = {})'.format(round(gini,3)))
    else:
        for i in range(len(fpr)):            
            plt.plot(fpr[i], tpr[i], label='ROC curve of {}. fold (GINI = {})'.format(i,round(gini[i],3)))
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of {} set'.format(label))
    plt.legend(loc="lower right")
    plt.show()



def LGB_CV(train_set,train_target,valid_set='',valid_target='',n_folds=3,ret_valid=0,cat_var='',use_timesplit=False):
    """
    Trains LGBM cross-validation
    """

    params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'subsample': 0.25,
            'subsample_freq': 1,
            'learning_rate': 0.4,
            'num_leaves': 20,
            'feature_fraction': 0.9,
            'lambda_l1': 1,  
            'lambda_l2': 1,
            'nthreads':3
            }
        
    data_cv=train_set.copy()
    target=train_target
    

    if valid_set is not None:
        data_valid=valid_set.copy()
        valid_predictions=np.zeros(data_valid.shape[0])   
        del valid_set
        
    del train_set
    gc.collect()
    

    if use_timesplit==False:
        folds=KFold(n_splits=n_folds)    
    else:
        folds=TimeSeriesSplit(n_splits=n_folds)
    
    iteration=1
    
    importance_df = pd.DataFrame()
    importance_df["Feature"] = list(data_cv)
    importance_df["importance_gain"]=0
    importance_df["importance_weight"]=0
    
    for train_index, test_index in folds.split(data_cv):
        X_train, X_test = data_cv.loc[train_index,:], data_cv.loc[test_index,:]
        y_train, y_test = target.loc[train_index], target.loc[test_index]
    
        dtrain = lgb.Dataset(X_train,label= y_train)
        dtest=lgb.Dataset(X_test, y_test)
        #watchlist = [(dtrain, 'train'), (dtest, 'test')]
        watchlist = dtest      
        
        gc.collect()
        
        booster = lgb.train(params,
                            dtrain,
                            valid_sets=watchlist,
                            early_stopping_rounds = 100,
                            num_boost_round = 100000,verbose_eval=200,
                            #categorical_feature=cat_var
                            )
        
        importance_df["importance_gain"] =importance_df["importance_gain"]+ booster.feature_importance(importance_type='gain')/n_folds
        importance_df["importance_weight"] =importance_df["importance_weight"]+ booster.feature_importance(importance_type='split')/n_folds
        

        try:
            data_valid
        except NameError:
            data_valid = None
        if data_valid is not None:
            valid_predictions=valid_predictions+np.expm1(booster.predict(data_valid))/n_folds
        
        iteration=iteration+1
        
    print('\n RESULTS: \n')    
    print('Number of observations in train sets is: {}'.format(X_train.shape[0]))
    print('Number of observations in test sets is: {}'.format(X_test.shape[0]))
    print('Average gini on train set:')
    print(round(np.mean(train_gini),4))    
    
    print('Average gini on test set:')
    print(round(np.mean(test_gini),4))   
    
            
    if ret_valid == 1:
        return importance_df,valid_predictions
            
    else:
        return importance_df  
    
    
    
def replace_categories(train_set,test_set,categorical_preds,num_categories):   
    """
    Merges categories of variables with more than num_categories categories
    and predits this transformation to test data
    """
    for i in categorical_preds:
        if train_set[i].nunique()>num_categories:
            print(i)
            print(train_set[i].nunique())
            top_n_cat=train_set[i].value_counts()[:10].index.tolist()
            train_set[i]=np.where(train_set[i].isin(top_n_cat),train_set[i],'other')   
            test_set[i]=np.where(test_trans[i].isin(top_n_cat),test_set[i],'other')
    return train_set,test_set
    

def compute_history(feature,hist_length,merged_weather):
    delka=hist_length*24
    
    grouped_weather=merged_weather #.groupby('site_id')
    name_of_feature_laged=feature+'_'+str(hist_length)
    merged_weather[name_of_feature_laged]=grouped_weather[feature].rolling(delka,min_periods=0).mean().reset_index(drop=True)
    merged_weather[name_of_feature_laged]=np.where(delka>merged_weather['site_rownum'],np.nan,merged_weather[name_of_feature_laged])    
    gc.collect()
        
    return merged_weather

    
    