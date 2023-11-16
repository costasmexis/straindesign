''' Generate synthetic data '''
import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, CopulaGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata

def synthetic_data_generation(df: pd.DataFrame, input: list, response: list, n_samples=100, embedding_dim=10, epochs=100, batch_size=10, verbose=True):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    ctgan = CTGANSynthesizer(metadata, batch_size=batch_size, epochs=epochs, embedding_dim=embedding_dim, verbose=verbose)
    ctgan.fit(df)
    data_gen = ctgan.sample(n_samples)
    X_gen = data_gen[input]
    y_gen = data_gen[response]
    print(f'Generated data shape: {data_gen.shape}')
    return data_gen, X_gen, y_gen

# grid.fit(X_gen, y_gen.values.ravel())
# model = grid.best_estimator_
# print(f'Best model: {model}')
# cv_on_whole_train_set(data, model)
 
# params_knn = {'n_neighbors': [2, 4, 6, 8], 'weights': ['uniform', 'distance']}
# params_svr = {'kernel': ['linear', 'rbf', 'sigmoid'], 'gamma': [0.001, 0.01, 0.05, 0.1, 1], 
#               'C': [5, 10, 20, 50, 100], 'epsilon': [0.001, 0.1, 0.2, 0.5, 1.0]}

# def model_validation(grid, X_train, y_train, X_val, y_val, model_name):
#     grid.fit(X_train, y_train)
#     model = grid.best_estimator_
#     y_pred = model.predict(X_val)
#     plot_pred_vs_actual(y_val, y_pred, model_name)
#     return model

# # grid = GridSearchCV(KNeighborsRegressor(n_jobs=-1), params_knn, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
# grid = GridSearchCV(SVR(), params_svr, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)

# # REAL data
# _ = model_validation(grid, X_train, y_train, X_val, y_val, 'Real Data')

# # SYNTHETIC data
# _ = model_validation(grid, X_gen, y_gen, X_val, y_val, 'Synthetic Data')

# # REAL + SYNTHETIC data
# X_train_gen = pd.concat([X_train, X_gen])
# y_train_gen = pd.concat([y_train, y_gen])
# _ = model_validation(grid, X_train_gen, y_train_gen, X_val, y_val, 'Real and Synthetic Data')

# # Validate on test set
# # REAL data
# _ = model_validation(grid, X_train, y_train, X_test, y_test, 'Real Data')

# # SYNTHETIC data
# _ = model_validation(grid, X_gen, y_gen, X_test, y_test, 'Synthetic Data')

# # REAL + SYNTHETIC data
# X_train_gen = pd.concat([X_train, X_gen])
# y_train_gen = pd.concat([y_train, y_gen])
# _ = model_validation(grid, X_train_gen, y_train_gen, X_test, y_test, 'Real and Synthetic Data')

# from table_evaluator import TableEvaluator

# data_gen['Limonene'] = y_gen['Limonene'].values

# table_evaluator = TableEvaluator(train, data_gen)
# table_evaluator.visual_evaluation()
# plot_corr_heatmap(data_gen)
# plot_corr_heatmap(train)

