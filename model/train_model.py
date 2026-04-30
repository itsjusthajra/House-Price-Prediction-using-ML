import sys
import os
import logging
import joblib
import numpy as np
import json
from xgboost import XGBRegressor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.preprocess import preprocess_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_model.pkl")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.json")


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    return {"r2": round(r2, 4), "mae": round(mae, 2), "mse": round(mse, 2), "rmse": round(rmse, 2)}


def get_feature_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "coef_"):
        return np.abs(model.coef_)
    return np.zeros(len(feature_names))


def train_all_models(data):
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42),
    }

    
    candidates["XGBoost"] = XGBRegressor(
            n_estimators=200, learning_rate=0.08, max_depth=5,
            random_state=42, verbosity=0, n_jobs=-1
    )

    results = {}

    for name, model in candidates.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = {"model": model, "metrics": metrics}
        logger.info(f"  R²={metrics['r2']:.4f} | MAE=₹{metrics['mae']:,.0f} | RMSE=₹{metrics['rmse']:,.0f}")

    return results


def select_best_model(results):
    best_name = max(results, key=lambda k: results[k]["metrics"]["r2"])
    logger.info(f"Best model: {best_name} (R²={results[best_name]['metrics']['r2']})")
    return best_name, results[best_name]["model"]


def save_artifacts(model, scaler, encoders, feature_names, metrics, model_results, best_model_name):
    artifact = {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names,
    }
    joblib.dump(artifact, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    serializable_results = {}
    for name, res in model_results.items():
        serializable_results[name] = {k: round(float(v), 4) for k, v in res["metrics"].items()}

    metadata = {
        "best_model": best_model_name,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "all_models": serializable_results,
        "feature_names": feature_names,
        "feature_importances": [round(float(x), 6) for x in get_feature_importances(model, feature_names)],
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {METADATA_PATH}")


def run_training(data_path="data/Housing.csv"):
    logger.info("Starting ML pipeline...")

    data = preprocess_pipeline(data_path)
    results = train_all_models(data)

    best_name, best_model = select_best_model(results)
    best_metrics = results[best_name]["metrics"]

    y_pred = best_model.predict(data["X_test"])

    save_artifacts(
        model=best_model,
        scaler=data["scaler"],
        encoders=data["encoders"],
        feature_names=data["feature_names"],
        metrics=best_metrics,
        model_results=results,
        best_model_name=best_name,
    )

    logger.info("Training complete.")
    return best_model, data, best_metrics, best_name, results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "Housing.csv")
    run_training(data_path)
