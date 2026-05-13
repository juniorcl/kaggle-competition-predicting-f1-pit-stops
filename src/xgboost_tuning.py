import os
import optuna
import pickle

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from feature_engine.selection import DropFeatures


os.environ["XGBOOST_VERBOSITY"] = "0"


def dump_pickle(file_obj, file_path):
    with open(file_path, 'bw') as file:
        pickle.dump(file_obj, file)


X_train = pd.read_parquet('../data/X_train.parquet')
y_train = pd.read_parquet('../data/y_train.parquet')


model = XGBClassifier(random_state=42, verbosity=0, enable_categorical=True).fit(X_train, y_train.PitNextLap)

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


scale_pos_weight = (y_train.PitNextLap == 0).sum() / (y_train.PitNextLap == 1).sum()

def objective(trial, X, y):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
        y_train, y_valid = y.iloc[train_idx, 0], y.iloc[valid_idx, 0]
        
        model = make_pipeline(
            DropFeatures(features_to_drop),
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                verbosity=0,
                scale_pos_weight=scale_pos_weight,
                enable_categorical=True,
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                n_estimators=trial.suggest_int("n_estimators", 100, 1500),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                gamma=trial.suggest_float("gamma", 0, 5),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
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


pipe_tuned = make_pipeline(
    DropFeatures(features_to_drop),
    XGBClassifier(
        enable_categorical=True,
        objective="binary:logistic",
        eval_metric="auc",
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
        **study.best_params
    )
).fit(X_train, y_train.PitNextLap)


dump_pickle(pipe_tuned, '../models/model_xgboost.pkl')