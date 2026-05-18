#%%
import sys
import optuna
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from feature_engine.encoding import RareLabelEncoder
from feature_engine.selection import DropFeatures


#%%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hist_gradient_boosting.log'), 
        logging.StreamHandler(sys.stdout)
    ]
)

optuna_logger = optuna.logging.get_logger("optuna")
optuna_logger.handlers = logging.getLogger().handlers
optuna_logger.setLevel(logging.INFO)

def dump_pickle(file_obj, file_path):
    with open(file_path, 'bw') as file:
        pickle.dump(file_obj, file)


#%%
X_train = pd.read_parquet('../data/processed/X_train.parquet')
y_train = pd.read_parquet('../data/processed/y_train.parquet')


#%%
logging.info("----- Feature Selection -----")

model = make_pipeline(
    RareLabelEncoder(variables=['driver']),
    HistGradientBoostingClassifier(class_weight='balanced', verbose=0),
).fit(X_train, y_train.PitNextLap)

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

logging.info(f"Features to drop: {features_to_drop}")


#%%
logging.info("----- Model Tuning -----")

def objective(trial, X, y):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
        y_train, y_valid = y.iloc[train_idx, 0], y.iloc[valid_idx, 0]

        model = make_pipeline(
            RareLabelEncoder(variables=['driver']),
            DropFeatures(features_to_drop),
            HistGradientBoostingClassifier(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 100),
                l2_regularization=trial.suggest_float("l2_regularization", 0.0, 10.0),
                max_bins=trial.suggest_int("max_bins", 64, 255),
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
                random_state=42,
                class_weight='balanced', 
                verbose=0
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
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50, n_jobs=-1, show_progress_bar=True)

logging.info(f"Best AUC: {study.best_value} | Best params: {study.best_params}")


#%%
logging.info("----- Saving Pipeline -----")

pipe_tuned = make_pipeline(
    RareLabelEncoder(variables=['driver']),
    DropFeatures(features_to_drop),
    HistGradientBoostingClassifier(
        **study.best_params,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42,
        class_weight='balanced',
        verbose=0
    )
).fit(X_train, y_train.PitNextLap)


dump_pickle(pipe_tuned, '../models/model_hist_gradient_boosting.pkl')

# %%
