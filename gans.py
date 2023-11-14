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
''' Counterfactual Explanation '''
import dice_ml
from dice_ml import Dice

d = dice_ml.Data(dataframe=data_A, continuous_features=INPUT_VARS, outcome_name='Limonene')
# We provide the type of model as a parameter (model_type)
m = dice_ml.Model(model=model, backend="sklearn", model_type='regressor')
exp_genetic = Dice(d, m, method="genetic")

# Multiple queries can be given as input at once
query_instances = X_train[2:4]
genetic_housing = exp_genetic.generate_counterfactuals(query_instances,
                                                               total_CFs=2,
                                                               desired_range=[3.0, 5.0])
genetic_housing.visualize_as_dataframe(show_only_changes=True)

# %%
''' OMLT '''
from omlt import OmltBlock, OffsetScaling
from omlt.io.keras import load_keras_sequential
from omlt.neuralnet import ReluBigMFormulation, FullSpaceSmoothNNFormulation
import pyomo.environ as pyo
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

dfin = X_train
dfout = y_train

inputs = INPUT_VARS
outputs = RESPONSE_VARS

x_offset, x_factor = dfin.mean().to_dict(), dfin.std().to_dict()
y_offset, y_factor = dfout.mean().to_dict(), dfout.std().to_dict()

dfin = (dfin - dfin.mean()).divide(dfin.std())
dfout = (dfout - dfout.mean()).divide(dfout.std())

x = dfin.values
y = dfout.values

# capture the minimum and maximum values of the scaled inputs
# so we don't use the model outside the valid range
scaled_lb = dfin.min()[inputs].values
scaled_ub = dfin.max()[inputs].values

# create our Keras Sequential model
nn = Sequential(name='ANN')
nn.add(Dense(units=516, input_dim=len(inputs), activation='relu'))
nn.add(Dense(1))
nn.compile(optimizer=Adam(), loss='mean_absolute_error', metrics=['mean_absolute_error'])

history = nn.fit(x, y, epochs=20)

# How to get predictions from trained ANN
# y_pred = nn.predict(x)
# y_pred = y_pred * y_factor['Limonene'] + y_offset['Limonene']

x_test = (X_test - x_offset).divide(x_factor)
x_test = np.array(x_test)
predictions = nn.predict(x_test)
predictions = predictions * y_factor['Limonene'] + y_offset['Limonene']
print(predictions)

# first, create the Pyomo model
m = pyo.ConcreteModel()
# create the OmltBlock to hold the neural network model
m.reformer = OmltBlock()
# load the Keras model
nn_reformer = nn

# Note: The neural network is in the scaled space. We want access to the
# variables in the unscaled space. Therefore, we need to tell OMLT about the
# scaling factors
scaler = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: y_offset[outputs[i]] for i in range(len(outputs))},
        factor_outputs={i: y_factor[outputs[i]] for i in range(len(outputs))}
    )

scaled_input_bounds = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}

# create a network definition from the Keras model
net = load_keras_sequential(nn_reformer, scaling_object=scaler, scaled_input_bounds=scaled_input_bounds)


# create the variables and constraints for the neural network in Pyomo
m.reformer.build_formulation(ReluBigMFormulation(net))

# now add the objective and the constraints
limonene_idx = outputs.index('Limonene')
m.obj = pyo.Objective(expr=m.reformer.outputs[limonene_idx], sense=pyo.maximize)

# now solve the optimization problem (this may take some time)
solver = pyo.SolverFactory('cplex')
status = solver.solve(m, tee=False)

for i in range(len(inputs)):
    print(f'{inputs[i]}:', pyo.value(m.reformer.inputs[i]))

print('Limonene: ', pyo.value(m.reformer.outputs[limonene_idx]))

# %%
''' SHAP '''
import shap

explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_train)

# Summary plot
shap.summary_plot(shap_values, X_train)
# Dependence plot
shap.dependence_plot('Q40322_MENSP', shap_values, X_train)

shap.initjs()

def shap_plot(j):
    explainerModel = shap.KernelExplainer(model.predict, X_train)
    shap_values_Model = explainerModel.shap_values(X_train)
    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], X_train.iloc[[j]])
    return p











# %%
''' Get sub-strain and validation data '''
train_size = int(0.8 * data_A.shape[0])
sub_train = data_A.iloc[:train_size, :]
val = data_A.iloc[train_size:, :]
X_sub_train = sub_train[INPUT_VARS]
y_sub_train = sub_train[RESPONSE_VARS]
X_val = val[INPUT_VARS]
y_val = val[RESPONSE_VARS]

# grid = GridSearchCV(KNeighborsRegressor(n_jobs=-1), params_knn, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid = GridSearchCV(SVR(), params_svr, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid.fit(X_sub_train, y_sub_train.values.ravel())
model = grid.best_estimator_
print(f'Best model: {model}')

y_pred = model.predict(X_val)
plot_pred_vs_actual(y_val, y_pred, 'Validation')

# %%
''' Generate synthetic data '''
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, CopulaGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data_A)

ctgan = CTGANSynthesizer(metadata, batch_size=10, epochs=100, 
                         embedding_dim=10, 
                         cuda=True, verbose=True)
ctgan.fit(data_A)

data_gen = ctgan.sample(30)
X_gen = data_gen[INPUT_VARS]
y_gen = data_gen[RESPONSE_VARS]

print(f'Generated data shape: {data_gen.shape}')

grid.fit(X_gen, y_gen.values.ravel())
model = grid.best_estimator_
print(f'Best model: {model}')
cv_on_whole_train_set(data, model)

# %%
params_knn = {'n_neighbors': [2, 4, 6, 8], 'weights': ['uniform', 'distance']}
params_svr = {'kernel': ['linear', 'rbf', 'sigmoid'], 'gamma': [0.001, 0.01, 0.05, 0.1, 1], 
              'C': [5, 10, 20, 50, 100], 'epsilon': [0.001, 0.1, 0.2, 0.5, 1.0]}

def model_validation(grid, X_train, y_train, X_val, y_val, model_name):
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    y_pred = model.predict(X_val)
    plot_pred_vs_actual(y_val, y_pred, model_name)
    return model

# grid = GridSearchCV(KNeighborsRegressor(n_jobs=-1), params_knn, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid = GridSearchCV(SVR(), params_svr, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)

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

data_gen['Limonene'] = y_gen['Limonene'].values

table_evaluator = TableEvaluator(train, data_gen)
table_evaluator.visual_evaluation()
plot_corr_heatmap(data_gen)
plot_corr_heatmap(train)

# %%
