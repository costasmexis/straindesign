#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from conf import *
from utils import *

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
# Split data_A to train & valdiation set
train_size = int(0.8*data_A.shape[0])
train = data_A.iloc[:train_size, :]
val = data_A.iloc[train_size:, :]
# Split train and val to X and y
X_train = train[INPUT_VARS]
y_train = train[RESPONSE_VARS]
X_val = val[INPUT_VARS]
y_val = val[RESPONSE_VARS]
# Split data_B to X and y test
X_test = data_B[INPUT_VARS]
y_test = data_B[RESPONSE_VARS]

# %% 
''' Train ML model on train set '''
params_knn = {'n_neighbors': [2, 4, 6, 8], 'weights': ['uniform', 'distance']}
grid = GridSearchCV(KNeighborsRegressor(n_jobs=-1), params_knn, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
plot_pred_vs_actual(y_val, y_pred, 'Accuracy on Validation')

# %%
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, CopulaGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=X_train)

ctgan = CTGANSynthesizer(metadata)
ctgan.fit(X_train)

gen = ctgan.sample(500)
X_gen = gen[INPUT_VARS]
y_gen = model.predict(X_gen)
y_gen = pd.DataFrame(y_gen, columns=y_train.columns)

# %%
params_knn = {'n_neighbors': [2, 4, 6, 8], 'weights': ['uniform', 'distance']}
params_svr = {'kernel': ['linear', 'rbf', 'sigmoid'], 'gamma': [0.001, 0.01, 0.1, 1], 'C': [0.1, 0.5, 1.0, 2.0, 5.0], 'epsilon': [0.1, 0.2, 0.5, 1.0]}

def model_validation(grid, X_train, y_train, X_val, y_val, model_name):
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    y_pred = model.predict(X_val)
    plot_pred_vs_actual(y_val, y_pred, model_name)
    return model

grid = GridSearchCV(KNeighborsRegressor(n_jobs=-1), params_knn, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
# grid = GridSearchCV(SVR(), params_svr, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)

# REAL data
_ = model_validation(grid, X_train, y_train, X_val, y_val, 'Real Data')

# SYNTHETIC data
_ = model_validation(grid, X_gen, y_gen, X_val, y_val, 'Synthetic Data')

# REAL + SYNTHETIC data
X_train_gen = pd.concat([X_train, X_gen])
y_train_gen = pd.concat([y_train, y_gen])
_ = model_validation(grid, X_train_gen, y_train_gen, X_val, y_val, 'Real and Synthetic Data')

# %%
# Validate on test set
# REAL data
_ = model_validation(grid, X_train, y_train, X_test, y_test, 'Real Data')

# SYNTHETIC data
_ = model_validation(grid, X_gen, y_gen, X_test, y_test, 'Synthetic Data')

# REAL + SYNTHETIC data
X_train_gen = pd.concat([X_train, X_gen])
y_train_gen = pd.concat([y_train, y_gen])
_ = model_validation(grid, X_train_gen, y_train_gen, X_test, y_test, 'Real and Synthetic Data')

# %%
from table_evaluator import TableEvaluator

table_evaluator = TableEvaluator(X_train, X_gen)
table_evaluator.visual_evaluation()
plot_corr_heatmap(gen)
plot_corr_heatmap(train)

# %%
