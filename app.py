import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocess import load_dataset, preprocess_pipeline
from utils.visualization import (
    price_distribution, area_vs_price, bedrooms_vs_price,
    bathrooms_vs_price, correlation_heatmap, feature_importance_chart,
    residual_plot, model_comparison_chart, feature_contribution_chart,
)
from model.train_model import (
    run_training, get_feature_importances, evaluate_model, MODEL_PATH, METADATA_PATH
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Housing.csv")


def load_or_train():
    if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
        logger.info("Loading saved model artifacts...")
        try:
            artifacts = joblib.load(MODEL_PATH)
            with open(METADATA_PATH) as f:
                metadata = json.load(f)
            return artifacts, metadata
        except Exception as e:
            logger.warning(f"Failed to load saved model: {e}. Retraining...")

    logger.info("No saved model found. Training now...")
    best_model, data, metrics, best_name, all_results = run_training(DATA_PATH)
    artifacts = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return artifacts, metadata


artifacts, metadata = load_or_train()
df_original = load_dataset(DATA_PATH)


@app.route("/")
def index():
    stats = {
        "total_houses": len(df_original),
        "avg_price": f"${df_original['price'].mean()/1e6:.2f}M",
        "max_price": f"${df_original['price'].max()/1e6:.2f}M",
        "min_price": f"${df_original['price'].min()/1e6:.2f}M",
        "avg_area": f"{df_original['area'].mean():,.0f} sq ft",
        "features": len(metadata["feature_names"]),
        "best_model": metadata["best_model"],
        "r2_score": f"{metadata['metrics']['r2']:.4f}",
        "mae": f"${metadata['metrics']['mae']/1e5:.2f}L",
        "rmse": f"${metadata['metrics']['rmse']/1e5:.2f}L",
    }
    return render_template("index.html", stats=stats, metadata=metadata)


@app.route("/dashboard")
def dashboard():
    charts = {
        "price_dist": price_distribution(df_original),
        "area_price": area_vs_price(df_original),
        "bedrooms_price": bedrooms_vs_price(df_original),
        "bathrooms_price": bathrooms_vs_price(df_original),
        "heatmap": correlation_heatmap(df_original),
    }

    importances = np.array(metadata["feature_importances"])
    feature_names = metadata["feature_names"]
    if importances.sum() > 0:
        charts["feature_importance"] = feature_importance_chart(feature_names, importances)

    charts["model_comparison"] = model_comparison_chart(metadata["all_models"])

    try:
        pipeline_data = preprocess_pipeline(DATA_PATH)
        y_pred = artifacts["model"].predict(pipeline_data["X_test"])
        charts["residuals"] = residual_plot(pipeline_data["y_test"], y_pred)
    except Exception as e:
        logger.warning(f"Residual plot skipped: {e}")

    return render_template("dashboard.html", charts=charts, metadata=metadata)


@app.route("/predict", methods=["GET"])
def predict_page():
    furnishing_options = df_original["furnishingstatus"].unique().tolist() if "furnishingstatus" in df_original.columns else ["furnished", "semi-furnished", "unfurnished"]
    return render_template("predict.html", furnishing_options=furnishing_options, metadata=metadata)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        area = float(data.get("area", 0))
        bedrooms = int(data.get("bedrooms", 2))
        bathrooms = int(data.get("bathrooms", 1))
        stories = int(data.get("stories", 1))
        parking = int(data.get("parking", 0))
        mainroad = data.get("mainroad", "yes")
        guestroom = data.get("guestroom", "no")
        basement = data.get("basement", "no")
        hotwaterheating = data.get("hotwaterheating", "no")
        airconditioning = data.get("airconditioning", "no")
        prefarea = data.get("prefarea", "no")
        furnishingstatus = data.get("furnishingstatus", "semi-furnished")

        binary_map = {"yes": 1, "no": 0}

        raw_input = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": binary_map.get(mainroad, 1),
            "guestroom": binary_map.get(guestroom, 0),
            "basement": binary_map.get(basement, 0),
            "hotwaterheating": binary_map.get(hotwaterheating, 0),
            "airconditioning": binary_map.get(airconditioning, 0),
            "parking": parking,
            "prefarea": binary_map.get(prefarea, 0),
        }

        # Handle furnishing one-hot (drop_first means "furnished" is baseline)
        raw_input["furnishingstatus_semi-furnished"] = 1 if furnishingstatus == "semi-furnished" else 0
        raw_input["furnishingstatus_unfurnished"] = 1 if furnishingstatus == "unfurnished" else 0

        feature_names = artifacts["feature_names"]
        input_vector = np.array([raw_input.get(f, 0) for f in feature_names]).reshape(1, -1)
        input_df = pd.DataFrame(input_vector, columns=feature_names)
        input_scaled = artifacts["scaler"].transform(input_df)

        predicted_price = float(artifacts["model"].predict(input_scaled)[0])

        importances = np.array(metadata["feature_importances"])
        contributions = importances * input_vector[0] * predicted_price / (importances.sum() + 1e-9)

        contribution_chart = feature_contribution_chart(feature_names, contributions)

        price_range_low = predicted_price * 0.92
        price_range_high = predicted_price * 1.08
        confidence = round(min(98, metadata["metrics"]["r2"] * 100 + 5), 1)

        return jsonify({
            "success": True,
            "predicted_price": predicted_price,
            "price_formatted": f"${predicted_price/1e6:.2f}M",
            "price_range": f"${price_range_low/1e5:.1f}L – ${price_range_high/1e5:.1f}L",
            "confidence": confidence,
            "model_used": metadata["best_model"],
            "r2": metadata["metrics"]["r2"],
            "contribution_chart": contribution_chart,
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/insights")
def insights():
    importances = np.array(metadata["feature_importances"])
    feature_names = metadata["feature_names"]

    importance_chart = None
    if importances.sum() > 0:
        importance_chart = feature_importance_chart(feature_names, importances)

    try:
        pipeline_data = preprocess_pipeline(DATA_PATH)
        y_pred = artifacts["model"].predict(pipeline_data["X_test"])
        residuals_chart = residual_plot(pipeline_data["y_test"], y_pred)
    except Exception:
        residuals_chart = None

    model_chart = model_comparison_chart(metadata["all_models"])

    paired = sorted(zip(importances, feature_names), reverse=True)[:8]
    top_features = [{"name": name, "score": round(score, 4)} for score, name in paired]

    return render_template(
        "insights.html",
        importance_chart=importance_chart,
        residuals_chart=residuals_chart,
        model_chart=model_chart,
        metadata=metadata,
        top_features=top_features,
    )


@app.route("/api/retrain", methods=["POST"])
def retrain():
    global artifacts, metadata
    try:
        run_training(DATA_PATH)
        artifacts = joblib.load(MODEL_PATH)
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        return jsonify({"success": True, "message": "Model retrained successfully", "r2": metadata["metrics"]["r2"]})
    except Exception as e:
        logger.error(f"Retrain error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
