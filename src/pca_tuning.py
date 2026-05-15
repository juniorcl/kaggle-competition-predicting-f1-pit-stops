#%%
import optuna
import pickle
import numpy as np
import pandas as pd

from sklearn import set_config

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler

from category_encoders import CatBoostEncoder


set_config(transform_output="pandas")


def dump_pickle(file_obj, file_path):
    with open(file_path, 'bw') as file:
        pickle.dump(file_obj, file)


#%%
column_transformer = ColumnTransformer([
    (
        'target_encoder', 
        TargetEncoder(), 
        ['driver', 'compound', 'race']
    ),
    (
        'catboost_encoder', 
        CatBoostEncoder(), 
        ['driver', 'compound', 'race']
    ),
    (
        'standard_scaler', 
        StandardScaler(), 
        ['lapnumber', 'position', 'raceprogress', 'year', 'position_norm', 'race_progress_sin', 'position_vs_mean']
    ),
    (
        'robust_scaler', 
        RobustScaler(), 
        [
            'position_change', 'cumulative_degradation', 'laptime_delta', 'laptime_s', 'stint', 'driver_mean_lap', 'tyrelife', 'delta_x_tyre_life', 
            'compound_tyre_life', 'stint_progress', 'tyre_life_ratio', 'degradation_per_lap', 'position_change_cum', 'laps_since_pit', 'lap_time_inv',  
            'lap_time_vs_race_mean', 'lap_time_x_tyre', 'position_x_progress', 'degradation_x_progress', 'race_progress_squared', 'driver_avg_position' 
        ]
    ),
], remainder="passthrough")


#%%
X_train = pd.read_parquet('../data/processed/X_train.parquet')
y_train = pd.read_parquet('../data/processed/y_train.parquet')


#%%
def objective(trial):

    pipe = make_pipeline(
        column_transformer,
        PCA(
            n_components=trial.suggest_float("n_components", 0.80, 0.99),
            svd_solver=trial.suggest_categorical("svd_solver", ["auto", "full"]),
            whiten=trial.suggest_categorical("whiten", [True, False]),
            iterated_power=trial.suggest_int("iterated_power", 1, 10),
            power_iteration_normalizer=trial.suggest_categorical("power_iteration_normalizer", ["auto", "QR", "LU"]),
        )
    )

    X_transformed = pipe.fit_transform(X_train, y_train.PitNextLap)

    explained_variance = np.sum(pipe.named_steps["pca"].explained_variance_ratio_)
    n_features = X_transformed.shape[1]

    return explained_variance - (0.001 * n_features)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300, n_jobs=-1, show_progress_bar=True)


print("Best Params:")
print(study.best_trial.params)

print("\nBest Score:")
print(study.best_trial.value)


#%%
pipe_column_transformer_pca = make_pipeline(
    column_transformer,
    PCA(**study.best_trial.params)
).fit(X_train, y_train.PitNextLap)


dump_pickle(pipe_column_transformer_pca, '../models/pipe_column_transformer_pca.pkl')
# %%
