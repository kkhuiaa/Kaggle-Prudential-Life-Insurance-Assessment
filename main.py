# -*- coding: utf-8 -*-
"""
The project is to develop a predictive model that accurately classifies risk using a more automated approach.
The target, "Response" is an ordinal variable relating to the final decision associated with an application.
The feature includes demographic info (age, weight, height, bmi) & insurance, family, medical history

Step1: Data Preparation
    Load the train, test data, define quadratic_weighted_kappa (Noted that our target is
Step2: Feature Engineering
    a. Create custom features by domain knowledge
    b. Transformation
        i. For categorical features, they will be transformed as ordinal variable (ranked by mean value versus the target variable) instead of dummy variable, avoiding the curse of dimensionality (where NA will be treated as single group)
        ii. For numerical features, the na will be assigned as -1
        iii. Standardize all the feature to standard score, which will be fastered in apply some models involving gradient descent
Step3: Recursive Feature Elimination a(RFE)
    Select a subset of data to try XGboost, RandomForest, LinearSVR models with Recursive Feature Elimination (RFE). In this process, we will understand the quadratic_weighted_kappa under each model and the relationship with the number of features. It is advised to read the documentation of RFE in sklearn.
Step4: Model Development
    a. Hyperparameter tuning
        XGboost algorithm is selected at the end of step3. Now we tune the hyperparameter of XGboost with a subset of data by stratified 5 folds of cross-validation.
    b. Final Model Tuning
        After the best set of Hyperparameter is selected, we will apply it to retrain the full set of data. In this part, we will slightly tune the parameter to avoid overfitting. We also test three approaches:
         - Regression approach with rounding (For example: regression value 1.34 -> 1)
         - Stacking approach (First apply Regression then a decision tree will assign the float value to the integer label)
         - multi-Classification approach
Step5: Model Evaluation
    a. Model is evaluated by stratified 5 folds of cross-validation, scoring with quadratic_weighted_kappa. Stacking approach is selected as final model.
    b. Confusion matrix is plotted by the training data
"""

#%%
# load packages
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import copy
import pickle as p
import itertools
import numpy as np , pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

from mlxtend.classifier import StackingClassifier
from xgboost import XGBRegressor, XGBClassifier

from ml_metrics import quadratic_weighted_kappa

from script import feature_selection_by_rfe, evaluation, categorytoordinal

def quadratic_weighted_kappa_round(estimator, X, actual):
    """This function applies the ml_metrics.quadratic_weighted_kappa without calling sklearn.metrics.make_scorer
    
    Parameters
    ----------
    estimator : estimator object implementing ‘fit’
        The object to use to fit the data.
    X : array-like
        The data to fit. Can be for example a list, or an array.
    actual : array-like
        the actual label
    
    Returns
    -------
    float
        ml_metrics.quadratic_weighted_kappa
    """
    predict = estimator.predict(X)
    unique_actual = list(set(actual))
    predict_round = [max(min(unique_actual), min(unique_actual, key=lambda x:abs(x-p))) for p in predict]
    return quadratic_weighted_kappa(actual, predict_round)

random_state = 1
target = "Response" #record the target here
other_threshold = 30

# load data
train0 = pd.read_csv('data/train.csv')
test0 = pd.read_csv('data/test.csv')
train0.loc[:, 'train_test'] = 'train'
test0.loc[:, 'train_test'] = 'test'
full_df = pd.concat([train0, test0], axis=0, sort=False)
print('train shape:', train0.shape) #train shape: (59381, 129)
print('test shape:', test0.shape) #test shape: (19765, 128)
print('full df shape:', full_df.shape) #full df shape: (79146, 129)
full_df.set_index('Id', inplace=True)

#create feature here
#since age, bmi, medical history are important in insurance (you will see it in Recursive Feature Elimination (RFE) session)
full_df.loc[:, 'Medical_Keyword_count'] = 0
for col in [col for col in full_df if 'Medical_Keyword_' in col]:
    full_df.loc[:, 'Medical_Keyword_count'] = full_df['Medical_Keyword_count'] + full_df[col].fillna(0)
full_df.loc[:, 'age_over_BMI'] = (full_df['Ins_Age']/full_df['BMI']).replace(np.inf, 9999)  #avoid there are some values too large
full_df.loc[:, 'age_time_BMI'] = full_df['Ins_Age']*full_df['BMI']
full_df.loc[:, 'Product_Info_2_char'] =  full_df.Product_Info_2.str[0]
full_df.loc[:, 'Product_Info_2_num'] =  full_df.Product_Info_2.str[1]
full_df.drop('Product_Info_2', axis=1, inplace=True)
full_df.loc[:, 'custom_var_1'] =  np.where(full_df['Medical_History_15'] < 10, 1, 0)
full_df.loc[:, 'custom_var_3'] =  np.where(full_df['Product_Info_4'] < 0.075, 1, 0)
full_df.loc[:, 'custom_var_4'] =  np.where(full_df['Product_Info_4'] == 1, 1, 0)
full_df.loc[:, 'custom_var_6'] = (full_df['BMI'] + 1)**2
full_df.loc[:, 'custom_var_7'] =  full_df['BMI']**0.8
full_df.loc[:, 'custom_var_8'] =  full_df['Ins_Age']**8.5
full_df.loc[:, 'BMI_Age'] = ( full_df['BMI']*full_df['Ins_Age'])**2.5
full_df.loc[:, 'custom_var_10'] =  np.where(full_df['BMI'] > np.percentile( full_df['BMI'], 0.8), 1, 0)
full_df.loc[:, 'custom_var_11'] = (full_df['BMI']*full_df['Product_Info_4'])**0.9
age_BMI_cutoff = np.percentile( full_df['BMI']*full_df['Ins_Age'], 0.9)
full_df.loc[:, 'custom_var_12'] = np.where(full_df['BMI']*full_df['Ins_Age']>age_BMI_cutoff, 1, 0)
full_df.loc[:, 'custom_var_13'] = (full_df['BMI']*full_df['Medical_Keyword_3'] + 0.5)**3

categorical =  ["Product_Info_1", "Product_Info_3", 
    "Product_Info_5", "Product_Info_6", "Product_Info_7", 
    "Employment_Info_2", "Employment_Info_3", "Employment_Info_5",
    "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3", 
    "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", 
    "InsuredInfo_7", "Insurance_History_1", "Insurance_History_2", 
    "Insurance_History_3", "Insurance_History_4", 
    "Insurance_History_7", "Insurance_History_8", 
    "Insurance_History_9", "Family_Hist_1", #"Medical_History_2", 
    "Medical_History_3", "Medical_History_4", "Medical_History_5", 
    "Medical_History_6", "Medical_History_7", "Medical_History_8", 
    "Medical_History_9",  "Medical_History_11", #"Medical_History_10",
    "Medical_History_12", "Medical_History_13", "Medical_History_14", 
    "Medical_History_16", "Medical_History_17", "Medical_History_18", 
    "Medical_History_19", "Medical_History_20", "Medical_History_21", 
    "Medical_History_22", "Medical_History_23", "Medical_History_25", 
    "Medical_History_26", "Medical_History_27", "Medical_History_28", 
    "Medical_History_29", "Medical_History_30", "Medical_History_31", 
    "Medical_History_33", "Medical_History_34", "Medical_History_35", 
    "Medical_History_36", "Medical_History_37", "Medical_History_38", 
    "Medical_History_39", "Medical_History_40", "Medical_History_41",
    'Product_Info_2_char', 'Product_Info_2_num']

train1 = full_df[full_df['train_test'] == 'train']

cto = categorytoordinal.CategoryToOrdinal(other_threshold=other_threshold)
cto.fit(train1[categorical], train1[target])
full_df1 = cto.transform(full_df)

full_df_nona = full_df1.fillna(-1)
sc = StandardScaler()
full_df_scalez_pre = full_df_nona.drop(['train_test', target], axis=1)
full_df_scalez = pd.DataFrame(data=sc.fit_transform(full_df_scalez_pre), index=full_df_scalez_pre.index, columns=full_df_scalez_pre.columns)

full_df_finished = full_df_scalez_pre.merge(full_df_nona[['train_test', target]], left_index=True, right_index=True)
print('merge before shape:', full_df_scalez_pre.shape) #merge before shape: (79146, 129)
print('merge after shape:', full_df_finished.shape) #merge after shape: (79146, 131)

train = full_df_finished[full_df_finished['train_test'] == 'train'].drop('train_test', axis=1)
test = full_df_finished[full_df_finished['train_test'] == 'test'].drop(['train_test', target], axis=1)
print('train shape:', train.shape) #train shape: (59381, 130)
print('finite test for train data:', np.all(np.isfinite(train))) #should be True
print('na test for train data', np.any(np.isnan(train))) #should be False

train_sample = train.sample(frac=.25, random_state=random_state)
X_rfe = train_sample.drop(target, axis=1)
y_rfe = train_sample[target]

#%%
#select a subset of data to faster the Recursive Feature Elimination (RFE)
feature_table_rfe = feature_selection_by_rfe.feature_selection_by_rfe(X_rfe, y_rfe
    , sklearn_model_list=[XGBRegressor, RandomForestRegressor, LinearSVR]
    , param_list=[{'n_estimators': 150, 'learning_rate': .1, 'min_child_weight': 50, 'subsample': .8, 'n_jobs': -1, 'reg_alpha': .3, 'colsample_bytree':.66}, None, None]
    , cv=3, n_jobs=-1, scoring=quadratic_weighted_kappa_round
    , random_state=random_state, plot_directory='evaluation/', plot_file_name='_'.join(['rfe', str(len(X_rfe.columns))]), show_plot=True)
print(feature_table_rfe[0])
#LinearSVR is not stable for too many features
#quadratic weighted kappa is low for RandomForest
#XGboost methdology is selected hence

#%%
with open(os.path.join('evaluation', 'feature_table_rfe_'+str(len(X_rfe.columns))+'_len_table'+'_seed_'+str(random_state)+'.pickle'), 'wb') as file:
    p.dump(feature_table_rfe, file)

#%%
# with open(os.path.join('evaluation', 'feature_table_rfe_'+str(len(X_rfe.columns))+'_len_table'+'_seed_'+str(random_state)+'.pickle'), 'rb') as file:
#     feature_table_rfe = p.load(file)
# selected_columns = feature_table_rfe[0][feature_table_rfe[0]['XGBRegressor_rank'] <= 70].index.tolist()
# pca = PCA(n_components=70)
# pca.fit(train[selected_columns])
# print('pca explained_variance_ratio_:', pca.explained_variance_ratio_.sum())
# X_train = pd.DataFrame(data=pca.transform(train[selected_columns]), index=train.index, columns=['pca_'+str(i+1) for i in range(0, pca.n_components_)])
# X_test = pd.DataFrame(data=pca.transform(test[selected_columns]), index=test.index, columns=['pca_'+str(i+1) for i in range(0, pca.n_components_)])
# y_train = train[train.index.isin(X_train.index)].reindex(X_train.index)[target]

# X_train = train[selected_columns]
# X_test = test[selected_columns]

X_train = train.drop(target, axis=1)
X_test = test.copy()
y_train = train[target]

print('X_train shape:', X_train.shape) #X_train shape: (59381, 30)
print('y_train shape:', y_train.shape) #y_train shape: (59381,)
print('X_test shape:', X_test.shape) #y_train shape: (59381,)

#%%
model_random_state = 2 #for tunning hyper-parameter only. In case we keep tunning the hyper-parameter, we should to change this random_state number
X_train_sample = X_train.sample(frac=.25, random_state=model_random_state) #select a subset of data to faster the hyper-parameter tunning process
y_train_sample = y_train[y_train.index.isin(X_train_sample.index)].reindex(X_train_sample.index)

param = {
    'learning_rate': [.1]
    , 'booster': ['gbtree', 'gblinear']
    , 'subsample': [.5, .6, .7]
    , 'n_estimators': [200]
    , 'min_child_weight': [.03]
    , 'colsample_bytree': [.6, .65, .7, .75, .8]
    , 'reg_alpha': [.2, .3, .4]
    , 'reg_lambda': [0]
    , 'max_depth': [3]
}
model = XGBRegressor(random_state=model_random_state, n_jobs=-1, early_stopping_rounds=80)
gridsearch = GridSearchCV(model, param_grid=param, cv=StratifiedKFold(5, random_state=model_random_state), n_jobs=1
    , scoring=quadratic_weighted_kappa_round)
t0 = time.time()
gridsearch.fit(X_train_sample, y_train_sample)
t1 = time.time()
print('time of gridsearch:', round(t1-t0, 2)) #calculate the time for gridsearch
print('best params:', gridsearch.best_params_)
best_position = gridsearch.best_index_
print('best train score:', gridsearch.cv_results_['mean_train_score'][best_position])
print('best train std:', gridsearch.cv_results_['std_train_score'][best_position])
print('best test score:', gridsearch.cv_results_['mean_test_score'][best_position])
print('best test std:', gridsearch.cv_results_['std_test_score'][best_position])

# best params: {'booster': 'gbtree', 'colsample_bytree': 0.75, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 0.03, 'n_estimators': 200, 'reg_alpha': 0.3, 'reg_lambda': 0, 'subsample': 0.5}
# best train score: 0.6411468942896711
# best train std: 0.003634929317673013
# best test score: 0.5849469590033762
# best test std: 0.007171962381064179

#%%
#after we have searched the best set of hyper-parameter, we now apply all the data to train the model with cross-validation
#we want to test for classification, regression or stacking, which one is the best.
cv = StratifiedKFold(5, random_state=model_random_state)
updated_dict = gridsearch.best_params_
updated_dict['learning_rate'] = .1
updated_dict['n_estimators'] = 800
updated_dict['min_child_weight'] = 50

#%%
print('===============XGboost regression with rounding===============')
xgbr = XGBRegressor(random_state=model_random_state, n_jobs=-1, early_stopping_rounds=80)
xgbr.set_params(**updated_dict)
xgbr_scores = evaluation.cv_scores(xgbr, X_train, y_train, cv=cv
    , scoring=quadratic_weighted_kappa_round, return_estimator=True)

# train mean of score: 0.6473931873941305
# train std of score: 0.001041262887225388
# test mean of score: 0.6063831053298916
# test std of score: 0.003201456042199307

#%%
print('===============XGboost regression with decision tree===============')
#It is very easy for Stacking to get overfitted, so we reduce the model complexity here
sclf = StackingClassifier(classifiers=[xgbr], meta_classifier=DecisionTreeClassifier(min_samples_leaf=500, random_state=model_random_state))
sclf_updated_dict = {'xgbregressor__'+k: v for k,v in updated_dict.items()}
sclf_updated_dict['xgbregressor__subsample'] = .4
sclf_updated_dict['xgbregressor__min_child_weight'] = 100
sclf.set_params(**sclf_updated_dict)
sclf_scores = evaluation.cv_scores(sclf, X_train, y_train, cv=cv
    , scoring=quadratic_weighted_kappa_round, return_estimator=True)

# train mean of score: 0.6434094278182996
# train std of score: 0.0021768938733617974
# test mean of score: 0.6069677840155301
# test std of score: 0.007928572700424638

#%%
print('===============XGboost classifiction with rounding===============')
xgbc = XGBClassifier(random_state=model_random_state, n_jobs=-1, early_stopping_rounds=80)
xgbc.set_params(**updated_dict)
xgbc_scores = evaluation.cv_scores(xgbc, X_train, y_train, cv=cv
    , scoring=quadratic_weighted_kappa_round, return_estimator=True)

# train mean of score: 0.595213394614275
# train std of score: 0.001755307149696452
# test mean of score: 0.5565338204660613
# test std of score: 0.00598958393061277

#==============================================================
#%%
#final model is stacking one
final_model = sclf_scores['estimator'][0] #choose final model here
p.dump(final_model, open(os.path.join('model', '_'.join([str(final_model).split('(')[0], 'seed_for_model',  str(model_random_state)+'_.pickle'])), 'wb')) #save the model

#%%
predict = final_model.predict(X_train)
train_kappa_score = quadratic_weighted_kappa(y_train, predict)
print(train_kappa_score) #0.6331364819900629
print(pd.Series(predict).value_counts()) 
# 8.0    24670
# 6.0    20645
# 2.0     7335
# 5.0     4146
# 1.0     1952
# 7.0      633
#%%
def plot_confusion_matrix(cm, classes, plot_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    matplotlib.rcParams.update({'font.size': 12})

    fmt = '.0%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    feature_selection_by_rfe.to_png(plt.gcf(), plot_path, show_plot=True, picture_size_h=9, picture_size_w=12)

cnf_matrix = confusion_matrix(y_train, predict)
np.set_printoptions(precision=1)

plot_confusion_matrix(cnf_matrix
    , plot_path='evaluation/confusion_matrix_without_normalization.png'
    , classes=list(set(y_train.astype(int))), title='Confusion matrix, without normalization'+'\n quadratic_weigh_kappa: '+str(round(train_kappa_score, 3)))
plot_confusion_matrix(cnf_matrix
    , plot_path='evaluation/confusion_matrix_with_normalization.png'
    , classes=list(set(y_train.astype(int))), normalize=True, title='Confusion matrix, with normalization'+'\n quadratic_weigh_kappa: '+str(round(train_kappa_score, 3)))

#%%
#submit the csv here
submission0 = X_test.copy()
submission0[target] = final_model.predict(submission0)
submission = submission0[[target]].reset_index()
submission.loc[:, target] = submission[target].astype(int)
print(submission[target].value_counts())
submission.to_csv('data/submission_sclf.csv', index=False)

#%%
# from stp_framework.config import Config
# from genlib.data import Data

# framework = Config(config_file="input/framework_config.json")
# # framework.create_framework()

# picture_size_scale = 2
# gendata = Data(train1)

# gendata.descriptive_statistics(gen_types=['feature', 'variable', 'NA'], numeric_param={'plot_directory' : framework.getdirectory('num_ds'), 'picture_size_scale': picture_size_scale}, category_param={'plot_directory' : framework.getdirectory('cate_ds'), 'picture_size_scale': picture_size_scale})
# for target in framework.config['target']:
#     gendata.update_gen_type(target, 'target')
#     gendata.bin_plots(gen_types=['feature', 'variable', 'NA'], numeric_param={'plot_directory' : framework.getdirectory('num_bin', target=target), 'picture_size_scale': picture_size_scale}, category_param={'plot_directory' : framework.getdirectory('cate_bin', target=target), 'min_obs': 5, 'picture_size_scale': picture_size_scale})
#     gendata.update_gen_type(target, 'variable')
