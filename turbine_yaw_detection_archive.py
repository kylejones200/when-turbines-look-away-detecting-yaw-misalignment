#!/usr/bin/env python3
"""
Yaw Misalignment Detection Using Čech Persistence
Detects turbine misalignment without yaw sensors using H2 void features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gudhi
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
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

def fetch_nrel_wind_data(lat=41.5, lon=-100.5, years=[2010, 2011, 2012]):
    """Simulate NREL Wind Toolkit data fetch."""
    logger.info(f"Simulating NREL wind data fetch for location ({lat}, {lon})")
    
    n_records = 365 * 24 * 12 * len(years)
    timestamps = pd.date_range(start=f'{years[0]}-01-01', periods=n_records, freq='5min')
    
    hours = np.array([t.hour + t.minute/60 for t in timestamps])
    days = np.array([t.dayofyear for t in timestamps])
    
    seasonal = 2 * np.sin(2 * np.pi * days / 365)
    diurnal = 1.5 * np.sin(2 * np.pi * hours / 24)
    
    windspeed_80m = 8.5 + seasonal + diurnal + np.random.normal(0, 2, n_records)
    windspeed_80m = np.clip(windspeed_80m, 0, 25)
    
    wind_direction = 180 + 60 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 15, n_records)
    wind_direction = wind_direction % 360
    
    temperature = 15 + 10 * np.cos(2 * np.pi * days / 365) + np.random.normal(0, 3, n_records)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'windspeed_80m': windspeed_80m,
        'wind_direction': wind_direction,
        'temperature': temperature
    })
    
    logger.info(f"Fetched {len(df)} records spanning {len(years)} years")
    return df

def simulate_turbine_power_yaw(windspeed, wind_direction, turbine_yaw, rated_power=2.0):
    """
    Simulate turbine power with yaw effects.
    Misalignment reduces power by cosine-cubed law.
    """
    cut_in = 3.0
    rated_speed = 12.0
    cut_out = 25.0
    
    power = np.zeros(len(windspeed))
    rotor_speed = np.zeros(len(windspeed))
    
    for i in range(len(windspeed)):
        ws = windspeed[i]
        
        if ws < cut_in or ws > cut_out:
            power[i] = 0
            rotor_speed[i] = 0
        else:
            # Base power curve
            base_power = np.where(ws < rated_speed, rated_power * ((ws - cut_in) / (rated_speed - cut_in)) ** 3, rated_power)
            
            # Yaw misalignment effect (cosine-cubed law)
            yaw_error = wind_direction[i] - turbine_yaw[i]
            yaw_error = ((yaw_error + 180) % 360) - 180  # Normalize to [-180, 180]
            yaw_factor = np.cos(np.radians(yaw_error)) ** 3
            yaw_factor = max(0, yaw_factor)
            
            power[i] = base_power * yaw_factor
            
            # Rotor speed proportional to wind speed
            rotor_speed[i] = 5 + (ws - cut_in) * 1.2
            
            # Add noise
            power[i] += np.random.normal(0, 0.03 * rated_power)
            rotor_speed[i] += np.random.normal(0, 0.5)
            
            power[i] = max(0, power[i])
            rotor_speed[i] = max(0, rotor_speed[i])
    
    return power, rotor_speed

def create_yaw_scenarios(df, n_windows=120, window_size=288):
    """
    Create labeled windows of aligned vs misaligned turbines.
    """
    logger.info(f"\nCreating {n_windows} labeled windows (aligned vs misaligned)...")
    
    windows = []
    labels = []
    
    max_start = len(df) - window_size
    starts = np.random.choice(max_start, n_windows, replace=False)
    
    for idx, start in enumerate(starts):
        window_df = df.iloc[start:start+window_size].copy()
        
        # Randomly assign misalignment (50/50 split)
        is_misaligned = (idx < n_windows // 2)
        
        if is_misaligned:
            # Misalignment: turbine yaw drifts from wind direction
            yaw_offset = np.random.uniform(5, 15)  # 5-15 degrees off
            if np.random.rand() > 0.5:
                yaw_offset = -yaw_offset
            turbine_yaw = window_df['wind_direction'].values + yaw_offset
        else:
            # Perfect alignment
            turbine_yaw = window_df['wind_direction'].values + np.random.normal(0, 2, len(window_df))
        
        power, rotor_speed = simulate_turbine_power_yaw(
            window_df['windspeed_80m'].values,
            window_df['wind_direction'].values,
            turbine_yaw
        )
        
        window_df['power'] = power
        window_df['rotor_speed'] = rotor_speed
        window_df['turbine_yaw'] = turbine_yaw
        
        windows.append(window_df)
        labels.append(1 if is_misaligned else 0)
    
    logger.info(f"Created {sum(labels)} misaligned windows and {len(labels)-sum(labels)} aligned windows")
    return windows, np.array(labels)

def compute_cech_persistence_features(window_df, max_dim=2):
    """
    Compute Čech persistence on 3D embedding (wind, power, rotor speed).
    Misalignment creates voids (H2 features).
    """
    windspeed = window_df['windspeed_80m'].values
    power = window_df['power'].values
    rotor_speed = window_df['rotor_speed'].values
    
    # Normalize to [0, 1]
    wind_norm = (windspeed - windspeed.min()) / (windspeed.max() - windspeed.min() + 1e-8)
    power_norm = (power - power.min()) / (power.max() - power.min() + 1e-8)
    rotor_norm = (rotor_speed - rotor_speed.min()) / (rotor_speed.max() - rotor_speed.min() + 1e-8)
    
    # Subsample to reduce computation
    n_samples = min(200, len(wind_norm))
    indices = np.random.choice(len(wind_norm), n_samples, replace=False)
    
    points = np.column_stack([wind_norm[indices], power_norm[indices], rotor_norm[indices]])
    
    # Compute Čech complex
    rips = gudhi.RipsComplex(points=points, max_edge_length=2.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim)
    simplex_tree.compute_persistence()
    
    features = {}
    
    for dim in range(max_dim + 1):
        persistence_pairs = simplex_tree.persistence_intervals_in_dimension(dim)
        
        if len(persistence_pairs) == 0:
            features[f'H{dim}_count'] = 0
            features[f'H{dim}_sum_lifetime'] = 0
            features[f'H{dim}_max_lifetime'] = 0
            features[f'H{dim}_mean_birth'] = 0
            features[f'H{dim}_mean_death'] = 0
        else:
            # Filter infinite lifetimes
            finite_pairs = persistence_pairs[np.isfinite(persistence_pairs).all(axis=1)]
            
            if len(finite_pairs) == 0:
                features[f'H{dim}_count'] = 0
                features[f'H{dim}_sum_lifetime'] = 0
                features[f'H{dim}_max_lifetime'] = 0
                features[f'H{dim}_mean_birth'] = 0
                features[f'H{dim}_mean_death'] = 0
            else:
                lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
                features[f'H{dim}_count'] = len(finite_pairs)
                features[f'H{dim}_sum_lifetime'] = np.sum(lifetimes)
                features[f'H{dim}_max_lifetime'] = np.max(lifetimes)
                features[f'H{dim}_mean_birth'] = np.mean(finite_pairs[:, 0])
                features[f'H{dim}_mean_death'] = np.mean(finite_pairs[:, 1])
    
    # Statistical features
    features['power_mean'] = power.mean()
    features['power_std'] = power.std()
    features['wind_mean'] = windspeed.mean()
    features['rotor_mean'] = rotor_speed.mean()
    features['power_wind_corr'] = np.corrcoef(power, windspeed)[0, 1]
    
    return features

def extract_all_features(windows, labels):
    """Extract Čech persistence features for all windows."""
    logger.info("\nExtracting Čech persistence features (H0, H1, H2)...")
    
    feature_list = []
    for i, window_df in enumerate(windows):
        if i % 20 == 0:
            logger.info(f"  Processing window {i+1}/{len(windows)}")
        
        features = compute_cech_persistence_features(window_df, max_dim=2)
        feature_list.append(features)
    
    X = pd.DataFrame(feature_list)
    y = labels
    
    logger.info(f"\nFeature matrix: {X.shape}")
    logger.info(f"Label distribution: Misaligned={sum(y)}, Aligned={len(y)-sum(y)}")
    
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
    
    return results, X_test, y_test

def generate_visualizations(windows, labels, X, y, results, out_dir):
    """Generate comprehensive visualizations."""
    logger.info("\n" + "="*60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*60)
    
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Model comparison
    logger.info("\n1. Model comparison...")
    fig, axes = plt.subplots(1, 2, figsize=tuple(config.get('output', {}).get('figsize', [12, 4])))
    
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    f1s = [results[m]['f1'] for m in model_names]
    
    axes[0].bar(range(len(model_names)), accuracies, color='#2b2b2b', alpha=0.85)
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Yaw Misalignment Detection Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    axes[1].bar(range(len(model_names)), f1s, color='#d62728', alpha=0.85)
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Yaw Misalignment Detection F1 Score')
    axes[1].set_ylim([0, 1])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: model_comparison.png")
    
    # 2. 3D phase space comparison
    logger.info("2. 3D phase space comparison...")
    fig = plt.figure(figsize=(14, 6))
    
    aligned_idx = np.where(labels == 0)[0][0]
    misaligned_idx = np.where(labels == 1)[0][0]
    
    aligned_window = windows[aligned_idx]
    misaligned_window = windows[misaligned_idx]
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(aligned_window['windspeed_80m'], aligned_window['power'], 
               aligned_window['rotor_speed'], alpha=0.5, s=10, color='#696969')
    ax1.set_xlabel('Wind Speed (m/s)')
    ax1.set_ylabel('Power (MW)')
    ax1.set_zlabel('Rotor Speed (RPM)')
    ax1.set_title('Aligned Turbine (compact structure', fontweight='normal)')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(misaligned_window['windspeed_80m'], misaligned_window['power'], 
               misaligned_window['rotor_speed'], alpha=0.5, s=10, color='#d62728')
    ax2.set_xlabel('Wind Speed (m/s)')
    ax2.set_ylabel('Power (MW)')
    ax2.set_zlabel('Rotor Speed (RPM)')
    ax2.set_title('Misaligned Turbine (void structure', fontweight='normal)')
    
    plt.tight_layout()
    plt.savefig(out_dir / "3d_phase_space.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: 3d_phase_space.png")
    
    # 3. H2 feature distributions
    logger.info("3. H2 feature distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    h2_features = ['H2_count', 'H2_max_lifetime', 'H2_sum_lifetime', 'H2_mean_birth']
    
    for ax, feature in zip(axes.flatten(), h2_features):
        aligned_values = X[y == 0][feature]
        misaligned_values = X[y == 1][feature]
        
        ax.hist(aligned_values, bins=20, alpha=0.6, label='Aligned', color='#696969', edgecolor='#2b2b2b')
        ax.hist(misaligned_values, bins=20, alpha=0.6, label='Misaligned', color='#d62728', edgecolor='#2b2b2b')
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution: {feature}')
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_dir / "h2_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: h2_distributions.png")
    
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
    logger.info("YAW MISALIGNMENT DETECTION USING ČECH PERSISTENCE")
    logger.info("="*60)
    
    df = fetch_nrel_wind_data()
    windows, labels = create_yaw_scenarios(df, n_windows=120, window_size=288)
    X, y = extract_all_features(windows, labels)
    results, X_test, y_test = train_and_evaluate_models(X, y)
    
    out_dir = Path(__file__).parent / "figures_yaw"
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

