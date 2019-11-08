import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import gc

from sklearn.model_selection import TimeSeriesSplit


def detect_types(data):
    numerical_preds=[]
    categorical_preds=[]
     
    
    for i in list(data):
        if(data[i].dtype=='object'):
            categorical_preds.append(i)
        else:
            numerical_preds.append(i)
    
    return numerical_preds,categorical_preds




def graph_exploration(feature_binned,target):
  
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
  
    if(sum(feature_binned.isnull())>0):
        feature_binned=feature_binned.cat.add_categories('NA').fillna('NA')
    
    plt.figure(figsize=(12,5))
    sns.boxplot(x=feature_binned,y=target,showfliers=False)
    plt.xticks(rotation='vertical')
    #plt.xlabel(feature_binned, fontsize=12)
    #plt.ylabel(target, fontsize=12)
    plt.show()
    



#USE_LGB_CV = True

def plot_imp(dataframe,imp_type,ret=False,n_predictors=100):
    plt.figure(figsize=(20,n_predictors/2))
    sns.barplot(x=imp_type, y="Feature", data=dataframe.sort_values(by=imp_type, ascending=False).head(n_predictors))
    plt.show()
    if ret==True:
        return list(dataframe.sort_values(by=imp_type, ascending=False).head(n_predictors)['Feature'])


def LGB_CV(train_set,train_target,valid_set='',valid_target='',n_folds=3,ret_valid=0,cat_var='',use_timesplit=False,obj=1,metrika=1):
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    
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
        
    #params={
    #        'learning_rate':0.05,
    #        'num_leaves':15,
    #        'colsample_bytree':0.75,
    #        'subsample':0.5,
    #        'subsample_freq':1,
    #        'max_depth':4,
    #        'nthreads':3,
    #        'verbose':1,
    #        'metric':'auc',
    #        'objective':'binary',
    #        'feature_name': 'auto' # that's actually the default
       #     'categorical_feature': cat_var # that's actually the defa
    #        }
    
    
    params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': obj,
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
         # 'learning_rate': 0.1,            
            "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric":metrika,
          "verbosity": 1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
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
    
    fpr_train=[]
    fpr_test=[]
    tpr_train=[]
    tpr_test=[]
    train_gini=[]
    test_gini=[] 

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
                            early_stopping_rounds = 200,
                            num_boost_round = 100000,verbose_eval=200,
                            #categorical_feature=cat_var
                            )
        
        importance_df["importance_gain"] =importance_df["importance_gain"]+ booster.feature_importance(importance_type='gain')/n_folds
        importance_df["importance_weight"] =importance_df["importance_weight"]+ booster.feature_importance(importance_type='split')/n_folds
        
        train_gini.append(2*roc_auc_score(y_train, booster.predict(X_train))-1)
        fpr, tpr, thr = roc_curve(y_train,booster.predict(X_train))
        fpr_train.append(fpr)
        tpr_train.append(tpr)
        print('Train Gini of fold {}'.format(iteration))
        print(train_gini[iteration-1])
        
        test_gini.append(2*roc_auc_score(y_test, booster.predict(X_test))-1)
        fpr, tpr, thr = roc_curve(y_test,booster.predict(X_test))
        fpr_test.append(fpr)
        tpr_test.append(tpr)
        print('Test Gini of fold {}'.format(iteration))
        print(test_gini[iteration-1])
        

        try:
            data_valid
        except NameError:
            data_valid = None
        if data_valid is not None:
            valid_predictions=valid_predictions+booster.predict(data_valid)/n_folds
        
        iteration=iteration+1
        
    print('\n RESULTS: \n')    
    print('Number of observations in train sets is: {}'.format(X_train.shape[0]))
    print('Number of observations in test sets is: {}'.format(X_test.shape[0]))
    print('Average gini on train set:')
    print(round(np.mean(train_gini),4))    
    
    print('Average gini on test set:')
    print(round(np.mean(test_gini),4))   
    
    if valid_target is not None:
        print('GINI on validation data (based on the mean prediction from particular lgb models):')
        valid_gini=2*roc_auc_score(valid_target,valid_predictions)-1
        print(round(valid_gini,4))
        fpr_valid, tpr_valid, thr = roc_curve(valid_target, valid_predictions)

    plot_roc(fpr_train,tpr_train,train_gini,'train')
    plot_roc(fpr_test,tpr_test,test_gini,'test')
            
    if valid_target is not None:
        plot_roc(fpr_valid,tpr_valid,valid_gini,'valid')
            
    if ret_valid==1:
        return importance_df,valid_predictions
            
    else:
        return importance_df  
    
    
    
def replace_categories(train_set,test_set,categorical_preds,num_categories):   
    for i in categorical_preds:
        if train_set[i].nunique()>num_categories:
            print(i)
            print(train_set[i].nunique())
            top_n_cat=train_set[i].value_counts()[:10].index.tolist()
            train_set[i]=np.where(train_set[i].isin(top_n_cat),train_set[i],'other')   
            test_set[i]=np.where(test_trans[i].isin(top_n_cat),test_set[i],'other')
    return train_set,test_set



def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def compute_history(feature,hist_length,merged_weather):
    delka=hist_length*24
    
    grouped_weather=merged_weather.groupby('site_id')
    #subset=merged_weather[merged_weather['site_id']==i].reset_index(drop=True)
    name_of_feature_laged=feature+'_'+str(delka)
    merged_weather[name_of_feature_laged]=grouped_weather[feature].rolling(hist_length,min_periods=0).mean().reset_index(drop=True)
    #grouped_weather=grouped_weather[['site_id','timestamp',name_of_feature_laged]]    
    #train=pd.merge(train,grouped_weather,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
    #test=pd.merge(test,grouped_weather,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])        
    gc.collect()
        
    return merged_weather

    
    