#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, \
    cross_validate, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils import *

PATH = 'limonene_data.csv'
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
data_b = data[data.index.isin(DBTL_B)] 

# %%
''' Plot correlation heatmap of DBTL 1st cycls'''
plot_corr_heatmap(data_A)

# %%
''' ML part '''
# Split data into input and response variables
X = data_A[INPUT_VARS]
y = data_A[RESPONSE_VARS]
# Define dictionary of estimators
estimators = {
    'LR': LinearRegression(),
    'DT': DecisionTreeRegressor(),
    'RF': RandomForestRegressor(),
    'SVR': SVR()
}
# Define parameters for grid search
params = {
    'LR': {
        'fit_intercept': [True, False]
    },
    'DT': {
        'max_depth': [2, 4, 6, 8, 10, 12, 14, 16],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
    },
    'RF': {
        'n_estimators': [10, 50, 100, 200, 500],
        'max_depth': [2, 4, 6, 8, 10, 12, 14, 16],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
    },
    'SVR': {
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': [0.001, 0.01, 0.1, 1],
        'C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'epsilon': [0.1, 0.2, 0.5, 1.0]
    }
}

def nested_cv(model, p_grid, X, y, n_trials = 10):
    nested_scores = []
    for i in tqdm(range(n_trials)):
        
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

        # Nested CV with parameter optimization
        clf = GridSearchCV(estimator=model, scoring='neg_mean_absolute_error', param_grid=p_grid, 
                                 cv=inner_cv)
        
        nested_score = cross_val_score(clf, X=X, y=y, 
                                       scoring='neg_mean_absolute_error', cv=outer_cv)
        
        nested_scores.append(list(nested_score))
    return clf, nested_scores

nested_LR, nested_scores_LR = nested_cv(estimators['LR'], params['LR'], X, y)
nested_DT, nested_scores_DT = nested_cv(estimators['DT'], params['DT'], X, y)
nested_RF, nested_scores_RF = nested_cv(estimators['RF'], params['RF'], X, y)
nested_SVR, nested_scores_SVR = nested_cv(estimators['SVR'], params['SVR'], X, y)

# %%
nested_scores_LR = [item for sublist in nested_scores_LR for item in sublist]
nested_scores_DT = [item for sublist in nested_scores_DT for item in sublist]
nested_scores_RF = [item for sublist in nested_scores_RF for item in sublist]
nested_scores_SVR = [item for sublist in nested_scores_SVR for item in sublist]

# create box plot
fig, ax = plt.subplots()
positions = [1, 2, 3, 4, 5] # x-axis positions for each box plot
ax.boxplot([nested_scores_LR, nested_scores_DT, nested_scores_RF, nested_scores_SVR], positions=positions, widths=0.1)
ax.set_title('Model Comparison', fontweight='bold')
ax.set_ylabel('MAE')
ax.set_xticks(positions)
ax.set_xticklabels(['LR', 'DT', 'RF', 'SVR'])
plt.show()
# %%
