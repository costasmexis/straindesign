#%%
''' Import libraries '''
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
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from tqdm import tqdm
from config import *
from utils import *
import warnings
warnings.filterwarnings("ignore")

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
# Reorder columns based on specific order
data_A = data_A[INPUT_VARS + RESPONSE_VARS]
data_B = data_B[INPUT_VARS + RESPONSE_VARS]

# %%
''' Data split '''
X_train = data_A[INPUT_VARS]
y_train = data_A[RESPONSE_VARS]
X_test = data_B[INPUT_VARS]
y_test = data_B[RESPONSE_VARS]

# %% 
''' Train ML models '''
def ml_training(X_train, y_train, param_grid, model):
    ''' Train ML model on train set '''
    grid = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train.values.ravel())
    model = grid.best_estimator_
    print(f'Best model: {model}')
    return model

params_knn = {'n_neighbors': [2, 4, 6, 8], 'weights': ['uniform', 'distance']}
params_svr = {'kernel': ['linear', 'rbf', 'sigmoid'], 'gamma': [0.001, 0.01, 0.05, 0.1, 1], 
              'C': [5, 10, 20, 50, 100, 150, 200], 'epsilon': [0.001, 0.1, 0.2, 0.5, 1.0]}

model = ml_training(X_train, y_train, params_svr, SVR())
cv_on_whole_train_set(data, model)

# %%
''' SHAPLEY VALUES '''
import shap
shap.initjs()

explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_train)

# Summary plot
shap.summary_plot(shap_values, X_train)

def shap_plot(j):
    explainerModel = shap.KernelExplainer(model.predict, X_train)
    shap_values_Model = explainerModel.shap_values(X_train)
    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], X_train.iloc[[j]])
    return p

# %%
''' Counterfactual Explanation '''
import dice_ml
from dice_ml import Dice

d = dice_ml.Data(dataframe=data_A, continuous_features=INPUT_VARS, outcome_name='Limonene')
m = dice_ml.Model(model=model, backend="sklearn", model_type='regressor')
exp_genetic = Dice(d, m, method="genetic")

# query_instances = X_train[2:4]
query_instances = pd.DataFrame(X_train.loc['B-Ml']).T
genetic_housing = exp_genetic.generate_counterfactuals(query_instances,
                                                               total_CFs=2,
                                                               desired_range=[3.0, 5.0])
genetic_housing.visualize_as_dataframe(show_only_changes=True)

# %%
# From X_train select specific instance based on index; Index is str type
query_instances = X_train.loc['B-Ml']
# %%
