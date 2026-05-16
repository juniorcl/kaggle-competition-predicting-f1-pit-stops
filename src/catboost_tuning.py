#%%
import optuna
import pickle

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
def dump_pickle(file_obj, file_path):
    with open(file_path, 'bw') as file:
        pickle.dump(file_obj, file)


#%%
X_train = pd.read_parquet('../data/X_train.parquet')
y_train = pd.read_parquet('../data/y_train.parquet')


#%%
model = make_pipeline(
    CatBoostEncoder(cols=['driver', 'compound', 'race']),
    CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced')
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


#%%
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
                iterations=5000,
                od_type="Iter",
                od_wait=200,
                random_state=42,
                verbose=0,
                thread_count=1,
                boosting_type=trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
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
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30, n_jobs=-1, show_progress_bar=True)

print(f"\nBest AUC: {study.best_trial.value:.6f}")

print("\nBest Params:")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")


#%%
pipe_tuned = make_pipeline(
    DropFeatures(features_to_drop),
    CatBoostEncoder(cols=['driver', 'compound', 'race']),
    CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=5000,
        od_type="Iter",
        od_wait=200,
        random_state=42,
        verbose=0,
        thread_count=1,
        **study.best_params
    )
).fit(X_train, y_train.PitNextLap)


dump_pickle(pipe_tuned, '../models/model_catboost.pkl')