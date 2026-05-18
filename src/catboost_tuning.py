#%%
import sys
import optuna
import pickle
import logging

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score

from feature_engine.selection import DropFeatures


#%%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/catboost.log'), 
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
    CatBoostEncoder(cols=['driver', 'compound', 'race']),
    CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced')
).fit(X_train, y_train.PitNextLap)

perm_result = permutation_importance(
    estimator=model, 
    X=X_train, 
    y=y_train.PitNextLap, 
    n_jobs=2, 
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

        X_train_fold = X.iloc[train_idx, :]
        X_valid_fold = X.iloc[valid_idx, :]

        y_train_fold = y.iloc[train_idx, 0]
        y_valid_fold = y.iloc[valid_idx, 0]

        model = make_pipeline(
            DropFeatures(features_to_drop),
            CatBoostEncoder(cols=['driver', 'compound', 'race']),
            CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                iterations=3000,
                od_type="Iter",
                od_wait=150,
                random_state=42,
                verbose=0,
                boosting_type=trial.suggest_categorical("boosting_type", ["Plain"]),
                depth=trial.suggest_int("depth", 4, 10),
                min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 100),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
                random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
                bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
                rsm=trial.suggest_float("rsm", 0.5, 1.0),
                auto_class_weights=trial.suggest_categorical("auto_class_weights", [None, "Balanced"]),
            )
        ).fit(X_train_fold, y_train_fold)

        proba = model.predict_proba(X_valid_fold)[:, 1]

        auc = roc_auc_score(y_valid_fold, proba)
        aucs.append(auc)

        print(f"Fold AUC: {auc:.6f}")

        trial.report(np.mean(aucs), step=fold)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(aucs)


study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30, n_jobs=2, show_progress_bar=True)

logging.info(f"Best AUC: {study.best_value} | Best params: {study.best_params}")


#%%
logging.info("----- Saving Pipeline -----")

pipe_tuned = make_pipeline(
    DropFeatures(features_to_drop),
    CatBoostEncoder(cols=['driver', 'compound', 'race']),
    CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=3000,
        od_type="Iter",
        od_wait=150,
        random_state=42,
        verbose=0,
        **study.best_params
    )
).fit(X_train, y_train.PitNextLap)


dump_pickle(pipe_tuned, '../models/model_catboost.pkl')