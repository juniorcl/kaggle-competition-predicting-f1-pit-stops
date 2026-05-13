import optuna
import pickle

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from feature_engine.selection import DropFeatures


def dump_pickle(file_obj, file_path):
    with open(file_path, 'bw') as file:
        pickle.dump(file_obj, file)


X_train = pd.read_parquet('../data/X_train.parquet')
y_train = pd.read_parquet('../data/y_train.parquet')


model = CatBoostClassifier(auto_class_weights='Balanced', verbose=0).fit(X_train, y_train.PitNextLap, cat_features=['driver', 'compound', 'race'])

perm_result = permutation_importance(
    estimator=model, 
    X=X_train, 
    y=y_train.PitNextLap, 
    n_jobs=-1, 
    scoring='roc_auc'
)

importance_df = pd.DataFrame({
    "feature": X_train.columns.tolist(),
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std
}).sort_values(by="importance_mean", ascending=False)

features_to_drop = importance_df.query("importance_mean <= 0").feature.tolist()


pipe_tuned = make_pipeline(
    DropFeatures(features_to_drop),
    CatBoostClassifier(auto_class_weights='Balanced', verbose=0)
).fit(X_train, y_train.PitNextLap, params={'cat_features': ['driver', 'compound', 'race']})


dump_pickle(pipe_tuned, '../models/model_catboost.pkl')