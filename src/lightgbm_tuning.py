#%%
import sys
import optuna
import pickle
import logging

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score

from feature_engine.encoding import RareLabelEncoder
from feature_engine.selection import DropFeatures


#%%
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('logs/lgbm.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def dump_pickle(file_obj, file_path):
    with open(file_path, 'bw') as file:
        pickle.dump(file_obj, file)


#%%
X_train = pd.read_parquet('../data/processed/X_train.parquet')
y_train = pd.read_parquet('../data/processed/y_train.parquet')


#%%
logger.info("----- Feature Selection -----")

model = LGBMClassifier(random_state=42, verbose=0, class_weight='balanced')
model.fit(X_train, y_train.PitNextLap)

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


#%%
logger.info("----- Fine Tuning -----")

def objective(trial, X, y):
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
        y_train, y_valid = y.iloc[train_idx, 0], y.iloc[valid_idx, 0]
        
        model = make_pipeline(
            DropFeatures(features_to_drop),
            LGBMClassifier(
                objective='binary',
                metric='auc',
                boosting_type='gbdt',
                num_leaves=trial.suggest_int('num_leaves', 16, 256),
                max_depth=trial.suggest_int('max_depth', 3, 12),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                lambda_l1=trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
                lambda_l2=trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
                feature_fraction=trial.suggest_float('feature_fraction', 0.6, 1.0),
                bagging_fraction=trial.suggest_float('bagging_fraction', 0.6, 1.0),
                bagging_freq=trial.suggest_int('bagging_freq', 1, 7),
                min_child_samples=trial.suggest_int('min_child_samples', 10, 100),
                verbosity=-1,
                n_estimators=2000,
                random_state=42,
                n_jobs=1
            )
        ).fit(X_train, y_train)

        proba = model.predict_proba(X_valid)[:, 1]
        
        auc = roc_auc_score(y_valid, proba)
        aucs.append(auc)

        trial.report(np.mean(aucs), step=fold)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(aucs)


study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30, n_jobs=-1, show_progress_bar=True)

logger.info(f"Best AUC: {study.best_value} | Best params: {study.best_params}")


#%%
logger.info("----- Saving Pipeline -----")

pipe_tuned = make_pipeline(
    DropFeatures(features_to_drop),
    LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        verbosity=-1,
        n_estimators=2000,
        random_state=42,
        **study.best_params
    )
).fit(X_train, y_train.PitNextLap)


dump_pickle(pipe_tuned, '../models/model_lightgbm.pkl')