"""
Build Plotly chart specs as plain Python dicts (no plotly package needed server-side).
The browser loads plotly from CDN and renders them directly.
"""

import numpy as np
import json


COLORS = {
    "primary": "#6C63FF",
    "secondary": "#FF6584",
    "accent": "#43E97B",
    "accent2": "#38F9D7",
    "muted": "#94a3b8",
    "text": "#e2e8f0",
}

LAYOUT_BASE = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#e2e8f0", "family": "Inter, sans-serif"},
    "margin": {"l": 48, "r": 16, "t": 48, "b": 40},
}

AXIS_STYLE = {"gridcolor": "rgba(255,255,255,0.05)", "zerolinecolor": "rgba(255,255,255,0.1)"}


def _layout(**overrides):
    layout = dict(LAYOUT_BASE)
    layout.update(overrides)
    return layout


def to_json(fig_dict):
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")
    return json.loads(json.dumps(fig_dict, default=_convert))


def price_distribution(df):
    prices = df["price"].dropna().tolist()
    fig = {
        "data": [{
            "type": "histogram",
            "x": prices,
            "nbinsx": 40,
            "marker": {"color": COLORS["primary"], "opacity": 0.85},
            "name": "Price",
        }],
        "layout": _layout(
            title={"text": "House Price Distribution", "font": {"size": 15}},
            xaxis={**AXIS_STYLE, "title": "Price (Rs)"},
            yaxis={**AXIS_STYLE, "title": "Count"},
            bargap=0.05,
        ),
    }
    return to_json(fig)


def area_vs_price(df):
    bed_col = df["bedrooms"].tolist() if "bedrooms" in df.columns else None
    fig = {
        "data": [{
            "type": "scatter",
            "x": df["area"].tolist(),
            "y": df["price"].tolist(),
            "mode": "markers",
            "marker": {
                "color": bed_col if bed_col else COLORS["primary"],
                "colorscale": "Viridis",
                "opacity": 0.65,
                "size": 6,
                "showscale": True if bed_col else False,
                "colorbar": {"title": "Beds", "thickness": 12},
            },
            "name": "Houses",
        }],
        "layout": _layout(
            title={"text": "Area vs Price", "font": {"size": 15}},
            xaxis={**AXIS_STYLE, "title": "Area (sq ft)"},
            yaxis={**AXIS_STYLE, "title": "Price (Rs)"},
        ),
    }
    return to_json(fig)


def bedrooms_vs_price(df):
    grouped = df.groupby("bedrooms")["price"].median().reset_index()
    fig = {
        "data": [{
            "type": "bar",
            "x": grouped["bedrooms"].tolist(),
            "y": grouped["price"].tolist(),
            "marker": {"color": COLORS["primary"], "opacity": 0.9},
            "text": [f"Rs{v/1e6:.1f}M" for v in grouped["price"]],
            "textposition": "outside",
        }],
        "layout": _layout(
            title={"text": "Median Price by Bedrooms", "font": {"size": 15}},
            xaxis={**AXIS_STYLE, "title": "Bedrooms"},
            yaxis={**AXIS_STYLE, "title": "Median Price (Rs)"},
            bargap=0.3,
        ),
    }
    return to_json(fig)


def bathrooms_vs_price(df):
    grouped = df.groupby("bathrooms")["price"].median().reset_index()
    fig = {
        "data": [{
            "type": "bar",
            "x": grouped["bathrooms"].tolist(),
            "y": grouped["price"].tolist(),
            "marker": {"color": COLORS["secondary"], "opacity": 0.9},
            "text": [f"Rs{v/1e6:.1f}M" for v in grouped["price"]],
            "textposition": "outside",
        }],
        "layout": _layout(
            title={"text": "Median Price by Bathrooms", "font": {"size": 15}},
            xaxis={**AXIS_STYLE, "title": "Bathrooms"},
            yaxis={**AXIS_STYLE, "title": "Median Price (Rs)"},
            bargap=0.3,
        ),
    }
    return to_json(fig)


def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    z = corr.values.tolist()
    cols = corr.columns.tolist()

    fig = {
        "data": [{
            "type": "heatmap",
            "z": z,
            "x": cols,
            "y": cols,
            "colorscale": "RdBu",
            "zmid": 0,
            "text": [[f"{v:.2f}" for v in row] for row in z],
            "texttemplate": "%{text}",
            "textfont": {"size": 10},
            "hoverongaps": False,
        }],
        "layout": _layout(
            title={"text": "Feature Correlation Matrix", "font": {"size": 15}},
            height=480,
        ),
    }
    return to_json(fig)


def feature_importance_chart(feature_names, importances):
    importances = np.array(importances)
    sorted_idx = np.argsort(importances).tolist()
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals = [float(importances[i]) for i in sorted_idx]

    fig = {
        "data": [{
            "type": "bar",
            "x": sorted_vals,
            "y": sorted_names,
            "orientation": "h",
            "marker": {
                "color": sorted_vals,
                "colorscale": "Viridis",
                "showscale": False,
            },
        }],
        "layout": _layout(
            title={"text": "Feature Importance", "font": {"size": 15}},
            xaxis={**AXIS_STYLE, "title": "Importance Score"},
            yaxis={**AXIS_STYLE},
            height=400,
        ),
    }
    return to_json(fig)


def residual_plot(y_true, y_pred):
    y_true_list = [float(v) for v in y_true]
    y_pred_list = [float(v) for v in y_pred]
    residuals = [a - b for a, b in zip(y_true_list, y_pred_list)]

    fig = {
        "data": [
            {
                "type": "scatter",
                "x": y_pred_list,
                "y": residuals,
                "mode": "markers",
                "marker": {"color": COLORS["primary"], "opacity": 0.55, "size": 6},
                "name": "Residuals",
                "xaxis": "x",
                "yaxis": "y",
            },
            {
                "type": "scatter",
                "x": [min(y_pred_list), max(y_pred_list)],
                "y": [0, 0],
                "mode": "lines",
                "line": {"color": COLORS["secondary"], "dash": "dash"},
                "name": "Zero line",
                "xaxis": "x",
                "yaxis": "y",
            },
            {
                "type": "histogram",
                "x": residuals,
                "nbinsx": 30,
                "marker": {"color": COLORS["accent"], "opacity": 0.8},
                "name": "Distribution",
                "xaxis": "x2",
                "yaxis": "y2",
            },
        ],
        "layout": _layout(
            title={"text": "Residual Analysis", "font": {"size": 15}},
            xaxis={"title": "Predicted", "domain": [0, 0.45], **AXIS_STYLE},
            yaxis={"title": "Residual", **AXIS_STYLE},
            xaxis2={"title": "Residual", "domain": [0.55, 1.0], "anchor": "y2", **AXIS_STYLE},
            yaxis2={"title": "Count", "anchor": "x2", **AXIS_STYLE},
            showlegend=False,
            height=380,
        ),
    }
    return to_json(fig)


def model_comparison_chart(model_results):
    models = list(model_results.keys())
    r2_scores = [float(model_results[m]["r2"]) for m in models]
    rmse_scores = [float(model_results[m]["rmse"]) for m in models]
    bar_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], "#FFD700"][:len(models)]

    fig = {
        "data": [
            {
                "type": "bar",
                "x": models,
                "y": r2_scores,
                "marker": {"color": bar_colors},
                "text": [f"{v:.3f}" for v in r2_scores],
                "textposition": "outside",
                "name": "R2",
                "xaxis": "x",
                "yaxis": "y",
            },
            {
                "type": "bar",
                "x": models,
                "y": rmse_scores,
                "marker": {"color": bar_colors},
                "text": [f"Rs{v/1e5:.1f}L" for v in rmse_scores],
                "textposition": "outside",
                "name": "RMSE",
                "xaxis": "x2",
                "yaxis": "y2",
            },
        ],
        "layout": _layout(
            title={"text": "Model Comparison", "font": {"size": 15}},
            xaxis={"domain": [0, 0.44], **AXIS_STYLE},
            yaxis={"title": "R2 Score", **AXIS_STYLE},
            xaxis2={"domain": [0.56, 1.0], "anchor": "y2", **AXIS_STYLE},
            yaxis2={"title": "RMSE (Rs)", "anchor": "x2", **AXIS_STYLE},
            showlegend=False,
            height=380,
            annotations=[
                {"text": "R2 Score (Higher = Better)", "x": 0.22, "y": 1.07,
                 "xref": "paper", "yref": "paper", "showarrow": False,
                 "font": {"size": 11, "color": COLORS["muted"]}},
                {"text": "RMSE (Lower = Better)", "x": 0.78, "y": 1.07,
                 "xref": "paper", "yref": "paper", "showarrow": False,
                 "font": {"size": 11, "color": COLORS["muted"]}},
            ],
        ),
    }
    return to_json(fig)


def feature_contribution_chart(feature_names, contributions):
    contributions = np.array(contributions)
    abs_idx = np.argsort(np.abs(contributions))[-10:]
    names = [feature_names[i] for i in abs_idx]
    vals = [float(contributions[i]) for i in abs_idx]
    colors = [COLORS["accent"] if v >= 0 else COLORS["secondary"] for v in vals]

    fig = {
        "data": [{
            "type": "bar",
            "x": vals,
            "y": names,
            "orientation": "h",
            "marker": {"color": colors},
            "text": [f"{v:+,.0f}" for v in vals],
            "textposition": "outside",
        }],
        "layout": _layout(
            title={"text": "Top Feature Contributions to Prediction", "font": {"size": 14}},
            xaxis={**AXIS_STYLE, "title": "Contribution (Rs)"},
            yaxis={**AXIS_STYLE},
            height=380,
        ),
    }
    return to_json(fig)
