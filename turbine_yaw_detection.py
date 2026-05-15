#!/usr/bin/env python3
"""
Yaw Misalignment Detection Using Čech Persistence
Detects turbine misalignment without yaw sensors using H2 void features.
"""

import logging
import os
from io import StringIO
from pathlib import Path
from typing import Any

import gudhi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# NREL API configuration (real data)
NREL_API_URL = (
    "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-bchrrr-v1-0-0-download.csv"
)
DEFAULT_NREL_EMAIL = "kyletjones@gmail.com"


def _get_nrel_api_key() -> str:
    """Return the NREL API key from the environment.

    Raises:
        RuntimeError: If the ``NREL_API_KEY`` environment variable is not set.
    """
    api_key = os.environ.get("NREL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NREL_API_KEY environment variable is not set. "
            "Export your key, e.g. `export NREL_API_KEY='your-key-here'`."
        )
    return api_key


def _normalize_nrel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize NREL Wind Toolkit column names to the ones used in this script."""
    cols = {c.lower(): c for c in df.columns}

    # Timestamp
    ts_col = None
    for key in ["timestamp", "time", "datetime", "date_time"]:
        if key in cols:
            ts_col = cols[key]
            break
    if ts_col is None:
        raise ValueError(
            f"Could not find a timestamp column in NREL data. Columns: {list(df.columns)}"
        )

    # Wind speed
    ws_col = None
    for key in ["wind_speed", "windspeed", "windspeed_80m", "wind_speed_80m"]:
        if key in cols:
            ws_col = cols[key]
            break
    if ws_col is None:
        raise ValueError("Could not find a wind speed column in NREL data.")

    # Wind direction
    wd_col = None
    for key in ["wind_direction", "winddir", "wind_direction_80m"]:
        if key in cols:
            wd_col = cols[key]
            break
    if wd_col is None:
        raise ValueError("Could not find a wind direction column in NREL data.")

    # Temperature
    temp_col = None
    for key in ["air_temperature", "temperature", "temperature_80m"]:
        if key in cols:
            temp_col = cols[key]
            break
    if temp_col is None:
        raise ValueError("Could not find a temperature column in NREL data.")

    df = df.rename(
        columns={
            ts_col: "timestamp",
            ws_col: "windspeed_80m",
            wd_col: "wind_direction",
            temp_col: "temperature",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["timestamp", "windspeed_80m", "wind_direction", "temperature"]]


def fetch_nrel_wind_data(
    lat: float = 41.5,
    lon: float = -100.5,
    years: list[int] | None = None,
    email: str = DEFAULT_NREL_EMAIL,
) -> pd.DataFrame:
    """Fetch real wind data from the NREL Wind Toolkit API."""
    if years is None:
        years = [2010, 2011, 2012]

    api_key = _get_nrel_api_key()
    logger.info(
        "Requesting NREL Wind Toolkit data lat=%.3f lon=%.3f years=%s", lat, lon, years
    )

    all_frames: list[pd.DataFrame] = []
    for year in years:
        params: dict[str, Any] = {
            "api_key": api_key,
            "lat": lat,
            "lon": lon,
            "year": year,
            "interval": 5,
            "email": email,
        }
        try:
            response = requests.get(NREL_API_URL, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error(
                "NREL request failed for year=%s: %s", year, exc, exc_info=True
            )
            raise RuntimeError(
                f"NREL API request failed for year {year}. See logs for details."
            ) from exc

        year_df = pd.read_csv(StringIO(response.text))
        year_df = _normalize_nrel_columns(year_df)
        pd.concat([all_frames, year_df])

    df = pd.concat(all_frames, axis=0).sort_values("timestamp").reset_index(drop=True)
    logger.info("Fetched %d NREL records spanning %d year(s)", len(df), len(years))
    return df


YAW_CUT_IN_SPEED_MPS = 3.0
YAW_RATED_SPEED_MPS = 12.0
YAW_CUT_OUT_SPEED_MPS = 25.0
YAW_RATED_POWER_MW = 2.0


def simulate_turbine_power_yaw(
    windspeed: np.ndarray,
    wind_direction: np.ndarray,
    turbine_yaw: np.ndarray,
    rated_power: float = YAW_RATED_POWER_MW,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate turbine power and rotor speed with yaw effects.

    Misalignment reduces power according to the cosine-cubed law.
    """
    power = np.zeros(len(windspeed))
    rotor_speed = np.zeros(len(windspeed))

    for i, ws in enumerate(windspeed):
        if ws < YAW_CUT_IN_SPEED_MPS or ws > YAW_CUT_OUT_SPEED_MPS:
            power[i] = 0.0
            rotor_speed[i] = 0.0
            continue

        if ws < YAW_RATED_SPEED_MPS:
            base_power = (
                rated_power
                * (
                    (ws - YAW_CUT_IN_SPEED_MPS)
                    / (YAW_RATED_SPEED_MPS - YAW_CUT_IN_SPEED_MPS)
                )
                ** 3
            )
        else:
            base_power = rated_power

        yaw_error = wind_direction[i] - turbine_yaw[i]
        yaw_error = ((yaw_error + 180) % 360) - 180  # Normalize to [-180, 180]
        yaw_factor = np.cos(np.radians(yaw_error)) ** 3
        yaw_factor = max(0.0, yaw_factor)

        power[i] = base_power * yaw_factor
        rotor_speed[i] = 5.0 + (ws - YAW_CUT_IN_SPEED_MPS) * 1.2

        power[i] += np.random.normal(0, 0.03 * rated_power)
        rotor_speed[i] += np.random.normal(0, 0.5)

        power[i] = max(0.0, power[i])
        rotor_speed[i] = max(0.0, rotor_speed[i])

    return power, rotor_speed


def create_yaw_scenarios(
    df: pd.DataFrame, n_windows: int = 120, window_size: int = 288
) -> tuple[list[pd.DataFrame], np.ndarray]:
    """Create labeled windows of aligned vs misaligned turbines."""
    logger.info("Creating %d labeled windows (aligned vs misaligned)", n_windows)

    windows = []
    labels = []

    max_start = len(df) - window_size
    starts = np.random.choice(max_start, n_windows, replace=False)

    for idx, start in enumerate(starts):
        window_df = df.iloc[start : start + window_size].copy()

        # Randomly assign misalignment (50/50 split)
        is_misaligned = idx < n_windows // 2

        if is_misaligned:
            # Misalignment: turbine yaw drifts from wind direction
            yaw_offset = np.random.uniform(5, 15)  # 5-15 degrees off
            if np.random.rand() > 0.5:
                yaw_offset = -yaw_offset
            turbine_yaw = window_df["wind_direction"].values + yaw_offset
        else:
            # Perfect alignment
            turbine_yaw = window_df["wind_direction"].values + np.random.normal(
                0, 2, len(window_df)
            )

        power, rotor_speed = simulate_turbine_power_yaw(
            window_df["windspeed_80m"].values,
            window_df["wind_direction"].values,
            turbine_yaw,
        )

        window_df["power"] = power
        window_df["rotor_speed"] = rotor_speed
        window_df["turbine_yaw"] = turbine_yaw

        pd.concat([windows, window_df])
        labels.append(1 if is_misaligned else 0)

    logger.info(
        "Created %d misaligned windows and %d aligned windows",
        int(sum(labels)),
        int(len(labels) - sum(labels)),
    )
    return windows, np.array(labels)


def compute_cech_persistence_features(window_df, max_dim=2):
    """
    Compute Čech persistence on 3D embedding (wind, power, rotor speed).
    Misalignment creates voids (H2 features).
    """
    windspeed = window_df["windspeed_80m"].values
    power = window_df["power"].values
    rotor_speed = window_df["rotor_speed"].values

    # Normalize to [0, 1]
    wind_norm = (windspeed - windspeed.min()) / (
        windspeed.max() - windspeed.min() + 1e-8
    )
    power_norm = (power - power.min()) / (power.max() - power.min() + 1e-8)
    rotor_norm = (rotor_speed - rotor_speed.min()) / (
        rotor_speed.max() - rotor_speed.min() + 1e-8
    )

    # Subsample to reduce computation
    n_samples = min(200, len(wind_norm))
    indices = np.random.choice(len(wind_norm), n_samples, replace=False)

    points = np.column_stack(
        [wind_norm[indices], power_norm[indices], rotor_norm[indices]]
    )

    # Compute Čech complex
    rips = gudhi.RipsComplex(points=points, max_edge_length=2.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim)
    simplex_tree.compute_persistence()

    features = {}

    for dim in range(max_dim + 1):
        persistence_pairs = simplex_tree.persistence_intervals_in_dimension(dim)

        if len(persistence_pairs) == 0:
            features[f"H{dim}_count"] = 0
            features[f"H{dim}_sum_lifetime"] = 0
            features[f"H{dim}_max_lifetime"] = 0
            features[f"H{dim}_mean_birth"] = 0
            features[f"H{dim}_mean_death"] = 0
        else:
            # Filter infinite lifetimes
            finite_pairs = persistence_pairs[np.isfinite(persistence_pairs).all(axis=1)]

            if len(finite_pairs) == 0:
                features[f"H{dim}_count"] = 0
                features[f"H{dim}_sum_lifetime"] = 0
                features[f"H{dim}_max_lifetime"] = 0
                features[f"H{dim}_mean_birth"] = 0
                features[f"H{dim}_mean_death"] = 0
            else:
                lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
                features[f"H{dim}_count"] = len(finite_pairs)
                features[f"H{dim}_sum_lifetime"] = np.sum(lifetimes)
                features[f"H{dim}_max_lifetime"] = np.max(lifetimes)
                features[f"H{dim}_mean_birth"] = np.mean(finite_pairs[:, 0])
                features[f"H{dim}_mean_death"] = np.mean(finite_pairs[:, 1])

    # Statistical features
    features["power_mean"] = power.mean()
    features["power_std"] = power.std()
    features["wind_mean"] = windspeed.mean()
    features["rotor_mean"] = rotor_speed.mean()
    features["power_wind_corr"] = np.corrcoef(power, windspeed)[0, 1]

    return features


def extract_all_features(
    windows: list[pd.DataFrame], labels: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract Čech persistence features for all windows."""
    logger.info("Extracting Čech persistence features (H0, H1, H2)")

    feature_list: list[dict[str, float]] = []
    for i, window_df in enumerate(windows):
        if i % 20 == 0:
            logger.info("Processing window %d/%d", i + 1, len(windows))

        features = compute_cech_persistence_features(window_df, max_dim=2)
        pd.concat([feature_list, features])

    X = pd.DataFrame(feature_list)
    y = labels

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    logger.info("Feature matrix shape: %s", X.shape)
    logger.info(
        "Label distribution: misaligned=%d aligned=%d",
        int(sum(y)),
        int(len(y) - sum(y)),
    )

    return X, y


def train_and_evaluate_models(
    X: pd.DataFrame, y: np.ndarray
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, np.ndarray]:
    """Train and evaluate classifiers."""
    logger.info("TRAINING AND EVALUATION")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logger.info("Train set: %d samples", len(X_train))
    logger.info("Test set: %d samples", len(X_test))

    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM (Linear)": SVC(kernel="linear", random_state=42, probability=True),
        "SVM (RBF)": SVC(kernel="rbf", random_state=42, probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    results: dict[str, dict[str, Any]] = {}

    for name, model in models.items():
        logger.info("Training model: %s", name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        logger.info(
            "  Accuracy=%.3f F1=%.3f AUC=%s", acc, f1, f"{auc:.3f}" if auc else "n/a"
        )

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    return results, X_test, y_test


def generate_visualizations(
    windows: list[pd.DataFrame],
    labels: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    results: dict[str, dict[str, Any]],
    out_dir: Path | str,
) -> None:
    """Generate and save all yaw misalignment visualizations."""
    logger.info("GENERATING VISUALIZATIONS")

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Model comparison
    logger.info("Creating model comparison plots")
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        model_names = list(results.keys())
        accuracies = [results[m]["accuracy"] for m in model_names]
        f1s = [results[m]["f1"] for m in model_names]

        axes[0].bar(range(len(model_names)), accuracies, color="#2b2b2b", alpha=0.85)
        axes[0].set_xticks(range(len(model_names)))
        axes[0].set_xticklabels(model_names, rotation=45, ha="right")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Yaw Misalignment Detection Accuracy")
        axes[0].set_ylim([0, 1])
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        axes[1].bar(range(len(model_names)), f1s, color="#d62728", alpha=0.85)
        axes[1].set_xticks(range(len(model_names)))
        axes[1].set_xticklabels(model_names, rotation=45, ha="right")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("Yaw Misalignment Detection F1 Score")
        axes[1].set_ylim([0, 1])
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(out_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved model comparison to %s", out_dir / "model_comparison.png")

        # 2. 3D phase space comparison
        logger.info("Creating 3D phase space comparison plots")
        fig = plt.figure(figsize=(14, 6))

        aligned_idx = np.where(labels == 0)[0][0]
        misaligned_idx = np.where(labels == 1)[0][0]

        aligned_window = windows[aligned_idx]
        misaligned_window = windows[misaligned_idx]

        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(
            aligned_window["windspeed_80m"],
            aligned_window["power"],
            aligned_window["rotor_speed"],
            alpha=0.5,
            s=10,
            color="#696969",
        )
        ax1.set_xlabel("Wind Speed (m/s)")
        ax1.set_ylabel("Power (MW)")
        ax1.set_zlabel("Rotor Speed (RPM)")
        ax1.set_title("Aligned Turbine (compact structure", fontweight="normal)")

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(
            misaligned_window["windspeed_80m"],
            misaligned_window["power"],
            misaligned_window["rotor_speed"],
            alpha=0.5,
            s=10,
            color="#d62728",
        )
        ax2.set_xlabel("Wind Speed (m/s)")
        ax2.set_ylabel("Power (MW)")
        ax2.set_zlabel("Rotor Speed (RPM)")
        ax2.set_title("Misaligned Turbine (void structure", fontweight="normal)")

        plt.tight_layout()
        plt.savefig(out_dir / "3d_phase_space.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(
            "Saved 3D phase space comparison to %s", out_dir / "3d_phase_space.png"
        )

        # 3. H2 feature distributions
        logger.info("Creating H2 feature distribution plots")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        h2_features = [
            "H2_count",
            "H2_max_lifetime",
            "H2_sum_lifetime",
            "H2_mean_birth",
        ]

        for ax, feature in zip(axes.flatten(), h2_features):
            aligned_values = X[y == 0][feature]
            misaligned_values = X[y == 1][feature]

            ax.hist(
                aligned_values,
                bins=20,
                alpha=0.6,
                label="Aligned",
                color="#696969",
                edgecolor="#2b2b2b",
            )
            ax.hist(
                misaligned_values,
                bins=20,
                alpha=0.6,
                label="Misaligned",
                color="#d62728",
                edgecolor="#2b2b2b",
            )
            ax.set_xlabel(feature)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution: {feature}")
            ax.legend(frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(out_dir / "h2_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved H2 distributions to %s", out_dir / "h2_distributions.png")

        # 4. Feature importance
        logger.info("Creating feature importance plot for Random Forest")
        if "Random Forest" in results:
            rf_model = results["Random Forest"]["model"]
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(
                range(len(indices)), importances[indices], color="#2b2b2b", alpha=0.85
            )
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels([X.columns[i] for i in indices], rotation=45, ha="right")
            ax.set_ylabel("Importance")
            ax.set_title("Top 10 Feature Importances (Random Forest)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            plt.savefig(
                out_dir / "feature_importance.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
        logger.info(
            "Saved feature importance to %s", out_dir / "feature_importance.png"
        )

    logger.info("All visualizations generated successfully")


def main() -> None:
    np.random.seed(42)
    """Main execution."""
    logger.info("YAW MISALIGNMENT DETECTION USING ČECH PERSISTENCE")

    df = fetch_nrel_wind_data()
    windows, labels = create_yaw_scenarios(df, n_windows=120, window_size=288)
    X, y = extract_all_features(windows, labels)
    results, X_test, y_test = train_and_evaluate_models(X, y)

    out_dir = Path(__file__).parent / "figures_yaw"
    generate_visualizations(windows, labels, X, y, results, out_dir)

    logger.info("FINAL SUMMARY")
    best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_result = results[best_model_name]
    logger.info(
        "Best model=%s accuracy=%.3f f1=%.3f",
        best_model_name,
        best_result["accuracy"],
        best_result["f1"],
    )
    logger.info("Visualizations saved to %s", out_dir)
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
