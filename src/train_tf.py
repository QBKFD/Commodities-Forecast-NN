import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import uuid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from src.shap_analysis import run_shap_analysis, summarize_shap_results
from src.architectures import (
    tf_stacked_gru,
    tf_stacked_lstm,
    tf_tft_lstm,
    tf_tft_gru,
    tf_transformers,
    benchmarks
)

MODEL_MAP = {
    "tf_stacked_gru": tf_stacked_gru,
    "tf_stacked_lstm": tf_stacked_lstm,
    "tf_hybrid_lstm": tf_tft_lstm,
    "tf_tft_gru": tf_tft_gru,
    "tf_transformers": tf_transformers,
    "benchmarks": benchmarks
}

def create_dual_sequences(X_price, X_macro, y, lookback, horizon=1):
    Xp, Xm, yt = [], [], []
    for i in range(lookback, len(X_price) - horizon + 1):
        Xp.append(X_price[i - lookback:i])
        Xm.append(X_macro[i - lookback:i])
        yt.append(y[i + horizon - 1])
    return np.array(Xp), np.array(Xm), np.array(yt)

def train_tf_model(model_name: str, config: dict):
    SEED = 42
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    run_id = str(uuid.uuid4())[:8]

    lookback = config["lookback"]
    H = config["horizon"]
    years = config["years"]

    df = pd.read_csv(config["dataset"], parse_dates=["ds"], thousands=',')
    df.set_index("ds", inplace=True)
    df.dropna(inplace=True)

    price_features = config["price_features"]
    macro_features = config["macro_features"]

    X_price_raw = df[price_features].values
    X_macro_raw = df[macro_features].values
    y = df["y"].values

    scaler_price = StandardScaler()
    scaler_macro = StandardScaler()
    scaler_target = StandardScaler()

    X_price_scaled = scaler_price.fit_transform(X_price_raw)
    X_macro_scaled = scaler_macro.fit_transform(X_macro_raw)
    y_scaled = scaler_target.fit_transform(y.reshape(-1, 1)).flatten()

    Xp_seq, Xm_seq, y_seq = create_dual_sequences(X_price_scaled, X_macro_scaled, y_scaled, lookback, H)
    dates_seq = df.index[lookback + H - 1 : lookback + H - 1 + len(y_seq)]

    n_price_features = Xp_seq.shape[2]
    n_macro_features = Xm_seq.shape[2]

    model_builder = MODEL_MAP[model_name]

    results = []
    all_preds = []
    shap_data = []

    os.makedirs("output/plots", exist_ok=True)
    os.makedirs("output/metrics", exist_ok=True)

    for year in years:
        print(f"\n Walk-forward for year {year}")

        train_mask = dates_seq < np.datetime64(f'{year}-01-01')
        test_mask = (dates_seq >= np.datetime64(f'{year}-01-01')) & (dates_seq < np.datetime64(f'{year + 1}-01-01'))

        Xp_train, Xm_train, y_train = Xp_seq[train_mask], Xm_seq[train_mask], y_seq[train_mask]
        Xp_test, Xm_test, y_test = Xp_seq[test_mask], Xm_seq[test_mask], y_seq[test_mask]

        if len(y_train) == 0 or len(y_test) == 0:
            print(f" Skipping {year} due to insufficient data.")
            continue

        print(f"{year} shapes — Xp_train: {Xp_train.shape}, Xm_train: {Xm_train.shape}, y_train: {y_train.shape}")

        if model_name == "benchmarks":
            model = model_builder.build_model(config)
            X_train_flat = np.concatenate([Xp_train, Xm_train], axis=2).reshape(Xp_train.shape[0], -1)
            X_test_flat = np.concatenate([Xp_test, Xm_test], axis=2).reshape(Xp_test.shape[0], -1)
            model.fit(X_train_flat, y_train)
            y_pred_scaled = model.predict(X_test_flat)
        else:
            model = model_builder.build_model(config, lookback, n_price_features, n_macro_features)

            callbacks = [
                EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-5)
            ]

            model.fit({'price_input': Xp_train, 'macro_input': Xm_train}, y_train,
                      epochs=config["epochs"],
                      batch_size=config["batch_size"],
                      verbose=0,
                      callbacks=callbacks)

            y_pred_scaled = model.predict({'price_input': Xp_test, 'macro_input': Xm_test}, verbose=0)

            shap_data.append({
                "year": year,
                "model": model,
                "Xp_train": Xp_train,
                "Xm_train": Xm_train,
                "Xp_test": Xp_test,
                "Xm_test": Xm_test,
                "y_test": y_test,
                "price_features": price_features,
                "macro_features": macro_features
            })

        y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_true = scaler_target.inverse_transform(y_test.reshape(-1, 1))

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        dir_acc = (np.sign(y_true.flatten()) == np.sign(y_pred.flatten())).mean()

        print(f" Year {year} — MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}, DA: {dir_acc:.2%}")
        results.append((year, mse, rmse, mae, r2, dir_acc))

        df_preds = pd.DataFrame({
            "ds": dates_seq[test_mask],
            "y_true": y_true.flatten(),
            "y_pred": y_pred.flatten(),
            "year": year
        })
        all_preds.append(df_preds)

        # Save per-year plot
        plt.figure(figsize=(14, 5))
        plt.plot(df_preds["ds"], df_preds["y_true"], label="Actual")
        plt.plot(df_preds["ds"], df_preds["y_pred"], label="Forecast", linestyle="--")
        plt.title(f"Rolling Forecast — {year}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"output/plots/forecast_{year}_{run_id}.png")
        plt.close()

    if all_preds:
        all_preds_df = pd.concat(all_preds, ignore_index=True)

        plt.figure(figsize=(18, 6))
        plt.plot(all_preds_df["ds"], all_preds_df["y_true"], label="Actual")
        plt.plot(all_preds_df["ds"], all_preds_df["y_pred"], label="Forecast", linestyle="--")
        plt.title("Rolling Forecast (All Years)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"output/plots/forecast_all_years_{run_id}.png")
        plt.close()

        results_df = pd.DataFrame(results, columns=["Year", "MSE", "RMSE", "MAE", "R2", "Directional Accuracy"])
        results_df.to_csv(f"output/metrics/metrics_by_year_{run_id}.csv", index=False)

        print("\n--- Average Metrics Across All Years ---")
        print(results_df.mean(numeric_only=True).round(4))

        if model_name != "benchmarks" and shap_data:
            run_shap = input("\nDo you want to run SHAP analysis? (yes/no): ").strip().lower()
            if run_shap in ["yes", "y"]:
                all_shap_values = []
                for item in shap_data:
                    run_shap_analysis(
                        item["model"],
                        item["Xp_train"],
                        item["Xm_train"],
                        item["Xp_test"],
                        item["Xm_test"],
                        item["y_test"],
                        item["price_features"],
                        item["macro_features"],
                        item["year"],
                        results,
                        years,
                        all_shap_values
                    )
                summarize_shap_results(all_shap_values, results, years)
            else:
                print("Skipping SHAP analysis.")
    else:
        print(" No predictions were generated. Check data and config.")
