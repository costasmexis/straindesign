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
from IPython.display import display
from conf import *
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
''' Basic data staticstics for the 2 cycles '''
print('DBTL 1st cycle')
display(data_A.describe())
print('DBTL 2nd cycle')
display(data_B.describe())

# Plot the distribution of every column of the 2 cycles; Use 4x3 grid plot
fig, axes = plt.subplots(4, 3, figsize=(20, 15))
for i, col in enumerate(data_A.columns):
    sns.histplot(data_A[col], ax=axes[i//3, i%3], bins=len(data_A), label='1st cycle')
    sns.histplot(data_B[col], ax=axes[i//3, i%3], bins=len(data_B), label='2nd cycle')
    fig.tight_layout(pad=3.0)
    axes[i//3, i%3].set_title(col)
    axes[i//3, i%3].legend()
plt.show()
