#!/usr/bin/env python3
"""
Icing Detection Using Multi-Parameter Persistence
Detects ice accumulation using persistence across temperature and time scales.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gudhi
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from pathlib import Path
import warnings
import logging
import yaml

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return {}
    with open(config_path) as _f:
        import yaml as _yaml
        return _yaml.safe_load(_f) or {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

np.random.seed(config.get('data', {}).get('seed', 42))

def fetch_nrel_wind_data(lat=45.0, lon=-95.0, years=[2010, 2011, 2012]):
    """Simulate NREL Wind Toolkit data fetch (northern location for icing)."""
    logger.info(f"Simulating NREL wind data fetch for location ({lat}, {lon})")
    
    n_records = 365 * 24 * 12 * len(years)
    timestamps = pd.date_range(start=f'{years[0]}-01-01', periods=n_records, freq='5min')
    
    hours = np.array([t.hour + t.minute/60 for t in timestamps])
    days = np.array([t.dayofyear for t in timestamps])
    
    seasonal = 2 * np.sin(2 * np.pi * days / 365)
    diurnal = 1.5 * np.sin(2 * np.pi * hours / 24)
    
    windspeed_80m = 8.5 + seasonal + diurnal + np.random.normal(0, 2, n_records)
    windspeed_80m = np.clip(windspeed_80m, 0, 25)
    
    # Temperature with realistic winter cold
    temperature = 5 + 15 * np.cos(2 * np.pi * days / 365 - np.pi/2) + np.random.normal(0, 4, n_records)
    temperature = np.clip(temperature, -25, 35)
    
    # Humidity (higher in winter)
    humidity = 65 + 20 * np.cos(2 * np.pi * days / 365 - np.pi/2) + np.random.normal(0, 10, n_records)
    humidity = np.clip(humidity, 30, 100)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'windspeed_80m': windspeed_80m,
        'temperature': temperature,
        'humidity': humidity
    })
    
    logger.info(f"Fetched {len(df)} records spanning {len(years)} years")
    return df

def simulate_turbine_power_icing(windspeed, temperature, humidity, icing_severity=0, rated_power=2.0):
    """
    Simulate turbine power with icing effects.
    icing_severity: 0 (no ice), 1 (light), 2 (moderate), 3 (severe)
    """
    cut_in = 3.0
    rated_speed = 12.0
    cut_out = 25.0
    
    # Icing reduces power by degrading aerodynamics
    ice_factors = [1.0, 0.85, 0.60, 0.30]
    ice_factor = ice_factors[icing_severity]
    
    power = np.zeros_like(windspeed)
    
    for i, ws in enumerate(windspeed):
        if ws < cut_in or ws > cut_out:
            power[i] = 0
        elif ws < rated_speed:
            power[i] = rated_power * ((ws - cut_in) / (rated_speed - cut_in)) ** 3
        else:
            power[i] = rated_power
        
        # Apply icing effect
        power[i] *= ice_factor
        
        # Ice adds additional variability
        noise_factor = 1.0 + icing_severity * 0.3
        power[i] += np.random.normal(0, 0.03 * rated_power * noise_factor)
        power[i] = max(0, power[i])
    
    return power

def create_icing_scenarios(df, n_windows=120, window_size=288):
    """
    Create labeled windows with and without icing.
    Icing occurs at T < 0°C and high humidity.
    """
    logger.info(f"\nCreating {n_windows} labeled windows (icing vs no-icing)...")
    
    windows = []
    labels = []
    
    max_start = len(df) - window_size
    
    # Find periods conducive to icing (cold + humid)
    icing_conditions = (df['temperature'] < 2) & (df['humidity'] > 75)
    no_icing_conditions = (df['temperature'] > 5) | (df['humidity'] < 60)
    
    icing_starts = np.where(icing_conditions)[0]
    no_icing_starts = np.where(no_icing_conditions)[0]
    
    # Filter valid starts
    icing_starts = icing_starts[(icing_starts < max_start)]
    no_icing_starts = no_icing_starts[(no_icing_starts < max_start)]
    
    # Sample equal numbers
    n_icing = n_windows // 2
    n_no_icing = n_windows - n_icing
    
    if len(icing_starts) < n_icing:
        logger.warning(f"  Warning: Only {len(icing_starts)} icing periods available, requested {n_icing}")
        n_icing = min(n_icing, len(icing_starts))
        n_no_icing = n_windows - n_icing
    
    if len(no_icing_starts) < n_no_icing:
        logger.warning(f"  Warning: Only {len(no_icing_starts)} no-icing periods available, requested {n_no_icing}")
        n_no_icing = min(n_no_icing, len(no_icing_starts))
    
    icing_sample = np.random.choice(icing_starts, n_icing, replace=False)
    no_icing_sample = np.random.choice(no_icing_starts, n_no_icing, replace=False)
    
    # Create icing windows
    for start in icing_sample:
        window_df = df.iloc[start:start+window_size].copy()
        
        icing_severity = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        
        power = simulate_turbine_power_icing(
            window_df['windspeed_80m'].values,
            window_df['temperature'].values,
            window_df['humidity'].values,
            icing_severity=icing_severity
        )
        
        window_df['power'] = power
        window_df['icing_severity'] = icing_severity
        
        windows.append(window_df)
        labels.append(1)
    
    # Create no-icing windows
    for start in no_icing_sample:
        window_df = df.iloc[start:start+window_size].copy()
        
        power = simulate_turbine_power_icing(
            window_df['windspeed_80m'].values,
            window_df['temperature'].values,
            window_df['humidity'].values,
            icing_severity=0
        )
        
        window_df['power'] = power
        window_df['icing_severity'] = 0
        
        windows.append(window_df)
        labels.append(0)
    
    logger.info(f"Created {sum(labels)} icing windows and {len(labels)-sum(labels)} no-icing windows")
    return windows, np.array(labels)

def compute_multiparam_persistence_features(window_df):
    """
    Compute multi-parameter persistence across temperature and time.
    Ice persists across both dimensions.
    """
    power = window_df['power'].values
    windspeed = window_df['windspeed_80m'].values
    temperature = window_df['temperature'].values
    
    # Normalize
    power_norm = (power - power.min()) / (power.max() - power.min() + 1e-8)
    wind_norm = (windspeed - windspeed.min()) / (windspeed.max() - windspeed.min() + 1e-8)
    temp_norm = (temperature - temperature.min()) / (temperature.max() - temperature.min() + 1e-8)
    
    # Subsample
    n_samples = min(150, len(power_norm))
    indices = np.random.choice(len(power_norm), n_samples, replace=False)
    indices = np.sort(indices)  # Keep temporal order
    
    points = np.column_stack([power_norm[indices], wind_norm[indices]])
    temps = temp_norm[indices]
    
    features = {}
    
    # Compute persistence at different temperature thresholds
    temp_thresholds = [0.3, 0.5, 0.7]
    
    for t_idx, thresh in enumerate(temp_thresholds):
        # Filter points by temperature
        mask = temps <= thresh
        if mask.sum() < 5:
            features[f'temp{t_idx}_H0_count'] = 0
            features[f'temp{t_idx}_H1_max'] = 0
            continue
        
        filtered_points = points[mask]
        
        # Compute Rips persistence
        rips = gudhi.RipsComplex(points=filtered_points, max_edge_length=2.0)
        simplex_tree = rips.create_simplex_tree(max_dimension=1)
        simplex_tree.compute_persistence()
        
        for dim in [0, 1]:
            persistence_pairs = simplex_tree.persistence_intervals_in_dimension(dim)
            finite_pairs = persistence_pairs[np.isfinite(persistence_pairs).all(axis=1)]
            
            if len(finite_pairs) > 0:
                lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
                features[f'temp{t_idx}_H{dim}_count'] = len(finite_pairs)
                features[f'temp{t_idx}_H{dim}_max'] = np.max(lifetimes)
                features[f'temp{t_idx}_H{dim}_mean'] = np.mean(lifetimes)
            else:
                features[f'temp{t_idx}_H{dim}_count'] = 0
                features[f'temp{t_idx}_H{dim}_max'] = 0
                features[f'temp{t_idx}_H{dim}_mean'] = 0
    
    # Time-scale persistence (different window sizes)
    time_windows = [50, 100, 150]
    
    for tw_idx, tw in enumerate(time_windows):
        if tw > len(points):
            features[f'time{tw_idx}_H1_max'] = 0
            continue
        
        sub_points = points[:tw]
        
        rips = gudhi.RipsComplex(points=sub_points, max_edge_length=2.0)
        simplex_tree = rips.create_simplex_tree(max_dimension=1)
        simplex_tree.compute_persistence()
        
        persistence_pairs = simplex_tree.persistence_intervals_in_dimension(1)
        finite_pairs = persistence_pairs[np.isfinite(persistence_pairs).all(axis=1)]
        
        if len(finite_pairs) > 0:
            lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
            features[f'time{tw_idx}_H1_max'] = np.max(lifetimes)
        else:
            features[f'time{tw_idx}_H1_max'] = 0
    
    # Statistical features
    features['power_mean'] = power.mean()
    features['power_std'] = power.std()
    features['temp_mean'] = temperature.mean()
    features['temp_min'] = temperature.min()
    features['wind_mean'] = windspeed.mean()
    features['power_cv'] = power.std() / (power.mean() + 1e-8)
    
    return features

def extract_all_features(windows, labels):
    """Extract multi-parameter persistence features."""
    logger.info("\nExtracting multi-parameter persistence features...")
    
    feature_list = []
    for i, window_df in enumerate(windows):
        if i % 20 == 0:
            logger.info(f"  Processing window {i+1}/{len(windows)}")
        
        features = compute_multiparam_persistence_features(window_df)
        feature_list.append(features)
    
    X = pd.DataFrame(feature_list)
    y = labels
    
    # Handle NaN and inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    logger.info(f"\nFeature matrix: {X.shape}")
    logger.info(f"Label distribution: Icing={sum(y)}, No-icing={len(y)-sum(y)}")
    
    return X, y

def train_and_evaluate_models(X, y):
    """Train and evaluate classifiers."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING AND EVALUATION")
    logger.info("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info(f"\nTrain set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\n{name}:")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        logger.info(f"  Accuracy: {acc:.3f}")
        logger.info(f"  F1 Score: {f1:.3f}")
        if auc is not None:
            logger.info(f"  AUC: {auc:.3f}")
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'f1': f1,
            'auc': auc,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    return results

def generate_visualizations(windows, labels, X, y, results, out_dir, plot: bool = False):
    """Generate comprehensive visualizations."""
    logger.info("\n" + "="*60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*60)
    
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Model comparison
    logger.info("\n1. Model comparison...")
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=tuple(config.get('output', {}).get('figsize', [12, 4])))
    
        model_names = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in model_names]
        f1s = [results[m]['f1'] for m in model_names]
    
        axes[0].bar(range(len(model_names)), accuracies, color='#2b2b2b', alpha=0.85)
        axes[0].set_xticks(range(len(model_names)))
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Icing Detection Accuracy')
        axes[0].set_ylim([0, 1])
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
    
        axes[1].bar(range(len(model_names)), f1s, color='#d62728', alpha=0.85)
        axes[1].set_xticks(range(len(model_names)))
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Icing Detection F1 Score')
        axes[1].set_ylim([0, 1])
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
    
        plt.tight_layout()
        plt.savefig(out_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: model_comparison.png")
    
    # 2. Temperature vs power scatter (icing vs no-icing)
        logger.info("2. Temperature-power relationship...")
        icing_idx = np.where(labels == 1)[0][0]
        no_icing_idx = np.where(labels == 0)[0][0]
    
        icing_window = windows[icing_idx]
        no_icing_window = windows[no_icing_idx]
    
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
        axes[0].scatter(no_icing_window['temperature'], no_icing_window['power'],
                       alpha=0.5, s=20, color='#696969', edgecolors='#2b2b2b', linewidth=0.5)
        axes[0].set_xlabel('Temperature (°C)')
        axes[0].set_ylabel('Power (MW)')
        axes[0].set_title('No Icing (normal operation)')
    
        axes[1].scatter(icing_window['temperature'], icing_window['power'],
                       alpha=0.5, s=20, color='#d62728', edgecolors='#8b0000', linewidth=0.5)
        axes[1].set_xlabel('Temperature (°C)')
        axes[1].set_ylabel('Power (MW)')
        axes[1].set_title('Icing Event (reduced power', fontweight='normal)')
    
        plt.tight_layout()
        plt.savefig(out_dir / "temperature_power.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: temperature_power.png")
    
    # 3. Multi-parameter feature distributions
        logger.info("3. Multi-parameter feature distributions...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
        mp_features = ['temp0_H1_max', 'temp1_H1_max', 'time0_H1_max', 'time1_H1_max']
    
        for ax, feature in zip(axes.flatten(), mp_features):
            if feature in X.columns:
                no_ice_values = X[y == 0][feature]
                ice_values = X[y == 1][feature]
            
                ax.hist(no_ice_values, bins=20, alpha=0.6, label='No Icing', color='#696969', edgecolor='#2b2b2b')
                ax.hist(ice_values, bins=20, alpha=0.6, label='Icing', color='#d62728', edgecolor='#2b2b2b')
                ax.set_xlabel(feature)
                ax.set_ylabel('Count')
                ax.set_title(f'Distribution: {feature}')
                ax.legend(frameon=False)
    
        plt.tight_layout()
        plt.savefig(out_dir / "multiparam_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: multiparam_distributions.png")
    
    # 4. Feature importance
        logger.info("4. Feature importance...")
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
        
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(indices)), importances[indices], color='#2b2b2b', alpha=0.85)
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels([X.columns[i] for i in indices], rotation=45, ha='right')
            ax.set_ylabel('Importance')
            ax.set_title('Top 10 Feature Importances (Random Forest)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(out_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
        logger.info(f"  Saved: feature_importance.png")
    
    logger.info("\nAll visualizations generated successfully!")

def main():
    """Main execution."""
    logger.info("="*60)
    logger.info("ICING DETECTION USING MULTI-PARAMETER PERSISTENCE")
    logger.info("="*60)
    
    df = fetch_nrel_wind_data()
    windows, labels = create_icing_scenarios(df, n_windows=120, window_size=288)
    X, y = extract_all_features(windows, labels)
    results = train_and_evaluate_models(X, y)
    
    out_dir = Path(__file__).parent / "figures_icing"
    generate_visualizations(windows, labels, X, y, results, out_dir)
    
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_result = results[best_model_name]
    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"  Accuracy: {best_result['accuracy']:.3f}")
    logger.info(f"  F1 Score: {best_result['f1']:.3f}")
    
    logger.info(f"\nVisualizations saved to: {out_dir}/")
    logger.info("\nAnalysis complete!")

if __name__ == "__main__":
    main()

