#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from tqdm import tqdm
from utils import *

PATH = 'data/limonene_data.csv'
MAX_ITER = 100000
RESPONSE_VARS = ['Limonene']
INPUT_VARS = ['ATOB_ECOLI','ERG8_YEAST','IDI_ECOLI',
                   'KIME_YEAST','MVD1_YEAST','Q40322_MENSP',
                   'Q8LKJ3_ABIGR','Q9FD86_STAAU','Q9FD87_STAAU']
DBTL_A = ['2X-Mh', 'B-Lm', '2X-Ll', 'A-Mm', 'B-Ll', 'A-Mh', '2X-Lm',
       'A-Hl', '2X-Hh', 'B-Ml', 'B-Mm', '2X-Lh', 'B-Mh', '2X-Hl', 'B-Hl',
       '2X-Ml', 'B-Hm', 'B-Lh', 'B-Hh', 'A-Ll', 'A-Hm', '2X-Mm', 'A-Hh',
       'A-Ml', 'A-Lm',  'A-Lh', '2X-Hm']
DBTL_B = ['BL-Mm', 'BL-Mh', 'BL-Ml']

# %%
''' Read original data and transform it for analysis'''
df = read_data(PATH)
print(f'Original data shape: {df.shape}')
data = transform_data(df)
print(f'Processed data shape: {data.shape}')
# DBTL 1st cycle 
data_A = data[data.index.isin(DBTL_A)]
# DBTL 2nd cycle
data_B = data[data.index.isin(DBTL_B)] 

# %%
''' Plot correlation heatmap of DBTL 1st cycls'''
plot_corr_heatmap(data_A)

# %%
''' ML part '''
# Split data into input and response variables
X = data_A[INPUT_VARS].values
y = data_A[RESPONSE_VARS].values.ravel()

# Define dictionary of estimators
estimators = {
    'LR': LinearRegression(),
    'RIDGE': Ridge(max_iter=MAX_ITER),
    'LASSO': Lasso(max_iter=MAX_ITER),
    'KNN': KNeighborsRegressor(),
    'MLP': MLPRegressor(max_iter=MAX_ITER),
    'DT': DecisionTreeRegressor(),
    'RF': RandomForestRegressor(),
    'SVR': SVR(),
    'XGB': XGBRegressor()
}

# Define parameters for grid search
params = {
    'LR': {
        'fit_intercept': [True, False]
    },
    'RIDGE': {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
    },
    'LASSO': {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
    },
    'KNN': {
        'n_neighbors': [2, 4, 6],
        'weights': ['uniform', 'distance'],
    },
    'MLP': {
        'hidden_layer_sizes': [(10,), (20,), (50,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.001, 0.01, 0.1],
        'learning_rate': ['adaptive', 'constant'],
    },
    'DT': {
        'max_depth': [2, 4, 6, 8, 10, 12, 14, 16],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
    },
    'RF': {
        'n_estimators': [10, 50, 100],
        'max_depth': [2, 4, 6, 8, 10, 12],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3],
    },
    'SVR': {
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': [0.001, 0.01, 0.1, 1],
        'C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'epsilon': [0.1, 0.2, 0.5, 1.0]
    },
    'XGB': {
        'n_estimators': [10, 20, 30],
        'max_depth': [2, 4, 6, 8, 10],
        'learning_rate': [0.001, 0.01, 0.1],
        'gamma': [0.001, 0.01, 0.1, 1],
    }
}

# %%
''' Nested cross-validation '''
nested_LR, nested_scores_LR = nested_cv(estimators['LR'], params['LR'], X, y)
nested_RIDGE, nested_scores_RIDGE = nested_cv(estimators['RIDGE'], params['RIDGE'], X, y)
nested_LASSO, nested_scores_LASSO = nested_cv(estimators['LASSO'], params['LASSO'], X, y)
nested_KNN, nested_scores_KNN = nested_cv(estimators['KNN'], params['KNN'], X, y)
nested_DT, nested_scores_DT = nested_cv(estimators['DT'], params['DT'], X, y)
nested_RF, nested_scores_RF = nested_cv(estimators['RF'], params['RF'], X, y)
nested_SVR, nested_scores_SVR = nested_cv(estimators['SVR'], params['SVR'], X, y)
nested_XGB, nested_scores_XGB = nested_cv(estimators['XGB'], params['XGB'], X, y)

# %%
# List of lists to list of elements
nested_scores_LR = [item for sublist in nested_scores_LR for item in sublist]
nested_scores_RIDGE = [item for sublist in nested_scores_RIDGE for item in sublist]
nested_scores_LASSO = [item for sublist in nested_scores_LASSO for item in sublist]
nested_scores_KNN = [item for sublist in nested_scores_KNN for item in sublist]
nested_scores_DT = [item for sublist in nested_scores_DT for item in sublist]
nested_scores_RF = [item for sublist in nested_scores_RF for item in sublist]
nested_scores_SVR = [item for sublist in nested_scores_SVR for item in sublist]
nested_scores_XGB = [item for sublist in nested_scores_XGB for item in sublist]
# Transform to positive
nested_scores_LR = [-x for x in nested_scores_LR]
nested_scores_RIDGE = [-x for x in nested_scores_RIDGE]
nested_scores_LASSO = [-x for x in nested_scores_LASSO]
nested_scores_KNN = [-x for x in nested_scores_KNN]
nested_scores_DT = [-x for x in nested_scores_DT]
nested_scores_RF = [-x for x in nested_scores_RF]
nested_scores_SVR = [-x for x in nested_scores_SVR]
nested_scores_XGB = [-x for x in nested_scores_XGB]

# Box plot of lists of scores
plt.boxplot([nested_scores_LR, nested_scores_RIDGE, nested_scores_LASSO, nested_scores_KNN, 
             nested_scores_DT, nested_scores_RF, nested_scores_SVR, nested_scores_XGB])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['LR', 'RIDGE', 'LASSO', 'KNN', 'DT', 'RF', 'SVR', 'XGB'])
plt.ylabel('MAE')
plt.savefig('plots/nested_cv.png', dpi=300)
plt.show()

# %%
''' Train a model on the whole dataset '''
X_train = data_A[INPUT_VARS].values
y_train = data_A[RESPONSE_VARS].values.ravel()
X_test = data_B[INPUT_VARS].values
y_test = data_B[RESPONSE_VARS].values.ravel()

# Tune and train a KNN model
knn = KNeighborsRegressor(n_jobs=-1)
params = {
    'n_neighbors': [2, 4, 6, 8],
    'weights': ['uniform', 'distance'],
}
grid = GridSearchCV(knn, params, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
knn = grid.best_estimator_
y_pred = cross_val_predict(knn, X_train, y_train, cv=5)
plot_pred_vs_actual(y_train, y_pred, 'KNN')

# Tune and train a SVR model
svr = SVR()
params = {
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': [0.001, 0.01, 0.1, 1],
    'C': [0.1, 0.5, 1.0, 2.0, 5.0],
    'epsilon': [0.1, 0.2, 0.5, 1.0]
}
grid = GridSearchCV(svr, params, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
svr = grid.best_estimator_
y_pred = cross_val_predict(svr, X_train, y_train, cv=5)
plot_pred_vs_actual(y_train, y_pred, 'SVR')

# %%
''' Test the model on the test set '''
# KNN
y_pred = knn.predict(X_test)
plot_pred_vs_actual(y_test, y_pred, 'KNN')
# SVR
y_pred = svr.predict(X_test)
plot_pred_vs_actual(y_test, y_pred, 'SVR')

# %%