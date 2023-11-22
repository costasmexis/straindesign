from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, matthews_corrcoef, \
        mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def bootstrap(
        estimator,
        param_grid,
        X,
        y,
        n_iterations=100,
        test_size=0.2,
        scoring = 'accuracy'):
    
    print(estimator.__class__.__name__)
    scorer = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'auc': roc_auc_score,
        'mcc': matthews_corrcoef,
        'mae': mean_absolute_error,
        'mse': mean_squared_error,
        'r2': r2_score
    }
    
    if scoring not in scorer.keys():
        raise ValueError(f'{scoring} not a valid scoring metric. \nPlease choose from {list(scorer.keys())}')

    evaluation_metrics = []

    for i in tqdm(range(n_iterations)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        grid = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        estimator = grid.best_estimator_
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        evaluation_metrics.append(scorer[scoring](y_test, y_pred))

    print(f'Average {scoring}: {np.mean(evaluation_metrics)}')
    print(f'Standard deviation {scoring}: {np.std(evaluation_metrics)}')

    return evaluation_metrics

def boxplot(data, labels)->None:
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=labels)
    plt.ylabel('Mean Error')
    plt.show()