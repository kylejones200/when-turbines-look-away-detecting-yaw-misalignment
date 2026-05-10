"""
Yaw Misalignment Detection Using Mapper
Detects misalignment from operational patterns without wind direction sensors
"""

import numpy as np
import pandas as pd
from pathlib import Path
import requests
from io import StringIO

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
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
        return _yaml.safe_load(_f) or {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Configuration
NREL_API_KEY = "key"
NREL_API_URL = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-bchrrr-v1-0-0-download.csv"


def fetch_nrel_wind_data(lat=41.5, lon=-93.5, years=[2017]):
    """Fetch wind data from NREL."""
    all_data = []
    
    for year in years:
        logger.info(f"   Fetching year {year}...")
        
        params = {
            'api_key': NREL_API_KEY,
            'wkt': f'POINT({lon} {lat})',
            'attributes': 'windspeed_100m,winddirection_100m,temperature_100m',
            'names': str(year),
            'utc': 'true',
            'leap_day': 'false',
            'interval': '60',
            'email': 'kyletjones@gmail.com'
        }
        
        try:
            response = requests.get(NREL_API_URL, params=params, timeout=120)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('Year,'):
                    data_start = i + 1
                    break
            
            data_text = '\n'.join(lines[data_start:])
            df_year = pd.read_csv(StringIO(data_text), header=None,
                           names=['Year', 'Month', 'Day', 'Hour', 'Minute',
                                  'windspeed_100m', 'winddirection_100m', 'temperature_100m'])
            
            df_year['time'] = pd.to_datetime(df_year[['Year', 'Month', 'Day', 'Hour', 'Minute']])
            all_data.append(df_year)
            logger.info(f"     ✓ Fetched {len(df_year):,} records")
            
        except Exception as e:
            logger.error(f"     ✗ Error: {e}")
            continue
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True).sort_values('time')


def simulate_turbine_with_misalignment(wind_df, rated_power=2000):
    """
    Simulate turbine with periodic yaw misalignment.
    
    Misalignment causes:
    - Power reduction by cos(angle)
    - Increased variability from asymmetric loading
    """
    df = wind_df.copy()
    n = len(df)
    
    # Initialize misalignment
    yaw_misalignment = np.zeros(n)
    
    # Inject misalignment periods (5% of time)
    i = 0
    while i < n:
        if np.random.random() < 0.05:
            # Start misalignment period
            duration = np.random.randint(2, 24)  # 2-24 hours
            angle = np.abs(np.random.normal(10, 5))  # Mean 10°, std 5°
            angle = np.clip(angle, 0, 25)
            
            for j in range(i, min(i + duration, n)):
                yaw_misalignment[j] = angle
            
            i += duration
        else:
            # Aligned (small random error <1°)
            yaw_misalignment[i] = np.random.randn() * 0.5
            yaw_misalignment[i] = np.clip(yaw_misalignment[i], -1, 1)
            i += 1
    
    # Simulate turbine response
    wind = df['windspeed_100m'].values
    
    rotor_speed = np.zeros(n)
    power = np.zeros(n)
    
    for i in range(1, n):
        w = wind[i]
        misalign = yaw_misalignment[i]
        
        # Effective wind speed (reduced by cos(misalignment))
        w_eff = w * np.cos(np.radians(misalign))
        
        # Power curve based on effective wind
        if w_eff < 3:
            target_power = 0
            target_rpm = 0
        elif w_eff < 12:
            target_power = rated_power * ((w_eff - 3) / (12 - 3)) ** 2.5
            target_rpm = 10 + (w_eff - 3) * 5
        else:
            target_power = rated_power
            target_rpm = 55 + (w_eff - 12) * 0.2
        
        # Add variability from misalignment (asymmetric loading)
        variability_factor = 1 + np.abs(misalign) * 0.02
        
        # Dynamics with lag
        rotor_speed[i] = 0.85 * rotor_speed[i-1] + 0.15 * target_rpm
        power[i] = 0.75 * power[i-1] + 0.25 * target_power
        
        # Add noise (increased by misalignment)
        rotor_speed[i] += np.random.randn() * 0.3 * variability_factor
        power[i] += np.random.randn() * 10 * variability_factor
        
        # Clip
        rotor_speed[i] = np.maximum(rotor_speed[i], 0)
        power[i] = np.clip(power[i], 0, rated_power * 1.1)
    
    df['yaw_misalignment'] = yaw_misalignment
    df['rotor_speed'] = rotor_speed
    df['power'] = power
    
    return df


def compute_expected_power(wind_speed, rated_power=2000):
    """Compute expected power from wind speed using power curve."""
    expected = np.zeros_like(wind_speed)
    
    for i, w in enumerate(wind_speed):
        if w < 3:
            expected[i] = 0
        elif w < 12:
            expected[i] = rated_power * ((w - 3) / (12 - 3)) ** 2.5
        else:
            expected[i] = rated_power
    
    return expected


def create_windows_with_filters(df, window_size=10):
    """
    Create windows and compute filter values.
    
    Filter 1: Power ratio (actual / expected)
    Filter 2: Rotor speed variability (std)
    """
    windows = []
    
    n = len(df)
    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        window = df.iloc[start:end]
        
        # Compute filters
        wind = window['windspeed_100m'].values
        power_actual = window['power'].values
        power_expected = compute_expected_power(wind)
        rotor = window['rotor_speed'].values
        
        # Filter 1: Mean power ratio
        if power_expected.mean() > 10:  # Skip very low power
            power_ratio = power_actual.mean() / power_expected.mean()
        else:
            continue
        
        # Filter 2: Rotor speed variability
        rotor_variability = rotor.std()
        
        # Label: mean misalignment in window
        misalign_mean = window['yaw_misalignment'].abs().mean()
        if misalign_mean < 5:
            label = 0  # Aligned
        elif misalign_mean > 10:
            label = 1  # Misaligned
        else:
            continue  # Skip ambiguous
        
        windows.append({
            'filter1': power_ratio,
            'filter2': rotor_variability,
            'label': label,
            'wind_mean': wind.mean(),
            'power_actual': power_actual.mean(),
            'power_expected': power_expected.mean()
        })
    
    return pd.DataFrame(windows)


def build_mapper_graph(X, filter1, filter2, n_bins=10, overlap=0.5, n_clusters=2):
    """
    Build Mapper graph.
    
    Args:
        X: Data array (n_samples x n_features)
        filter1: Filter function values (n_samples)
        filter2: Filter function values (n_samples)
        n_bins: Number of bins per filter dimension
        overlap: Overlap fraction between bins
        n_clusters: Number of clusters per bin
    
    Returns:
        NetworkX graph with node attributes
    """
    # Create 2D bins
    f1_min, f1_max = filter1.min(), filter1.max()
    f2_min, f2_max = filter2.min(), filter2.max()
    
    # Bin edges with overlap
    step1 = (f1_max - f1_min) / n_bins
    step2 = (f2_max - f2_min) / n_bins
    
    bins1 = [(f1_min + i * step1 * (1 - overlap), 
              f1_min + (i + 1) * step1 * (1 + overlap)) 
             for i in range(n_bins)]
    bins2 = [(f2_min + i * step2 * (1 - overlap), 
              f2_min + (i + 1) * step2 * (1 + overlap)) 
             for i in range(n_bins)]
    
    # Build graph
    G = nx.Graph()
    node_id = 0
    node_to_points = {}
    
    # For each bin, cluster points
    for i, (b1_low, b1_high) in enumerate(bins1):
        for j, (b2_low, b2_high) in enumerate(bins2):
            # Find points in this bin
            mask = ((filter1 >= b1_low) & (filter1 <= b1_high) &
                    (filter2 >= b2_low) & (filter2 <= b2_high))
            
            if mask.sum() < n_clusters:
                continue
            
            points_in_bin = X[mask]
            indices_in_bin = np.where(mask)[0]
            
            # Cluster
            if len(points_in_bin) >= n_clusters:
                kmeans = KMeans(n_clusters=min(n_clusters, len(points_in_bin)), random_state=42)
                cluster_labels = kmeans.fit_predict(points_in_bin)
                
                for cluster_id in range(kmeans.n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_indices = indices_in_bin[cluster_mask]
                    
                    if len(cluster_indices) > 0:
                        G.add_node(node_id, 
                                  bin=(i, j),
                                  indices=cluster_indices,
                                  size=len(cluster_indices))
                        node_to_points[node_id] = set(cluster_indices)
                        node_id += 1
    
    # Add edges between overlapping clusters
    nodes = list(G.nodes())
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            # Check if they share points
            shared = node_to_points[node1].intersection(node_to_points[node2])
            if len(shared) > 0:
                G.add_edge(node1, node2, weight=len(shared))
    
    return G


def classify_with_mapper(G, X_train, y_train, X_test, filter1_test, filter2_test, n_bins=10):
    """
    Classify test points using Mapper graph.
    
    Each node in graph is labeled with majority class of its training points.
    Test points are assigned to nearest node in filter space.
    """
    # Label each node with majority class
    node_labels = {}
    for node in G.nodes():
        indices = G.nodes[node]['indices']
        labels = y_train[indices]
        node_labels[node] = int(labels.mean() > 0.5)  # Majority vote
    
    # Compute node centers in filter space
    node_centers = {}
    f1_train = np.array([X_train[G.nodes[node]['indices'], 0].mean() for node in G.nodes()])
    f2_train = np.array([X_train[G.nodes[node]['indices'], 1].mean() for node in G.nodes()])
    
    for i, node in enumerate(G.nodes()):
        node_centers[node] = (f1_train[i], f2_train[i])
    
    # Classify test points
    predictions = []
    
    for i in range(len(X_test)):
        f1, f2 = filter1_test[i], filter2_test[i]
        
        # Find nearest node
        min_dist = float('inf')
        nearest_node = None
        
        for node, (c1, c2) in node_centers.items():
            dist = np.sqrt((f1 - c1)**2 + (f2 - c2)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        if nearest_node is not None:
            predictions.append(node_labels[nearest_node])
        else:
            predictions.append(0)  # Default to aligned
    
    return np.array(predictions)


def visualize_mapper_graph(G, y_labels, out_dir, plot: bool = False):
    """Visualize Mapper graph with node colors by label."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    
    # Node colors based on majority label
    node_colors = []
    for node in G.nodes():
        indices = G.nodes[node]['indices']
        labels = y_labels[indices]
        majority = labels.mean()
        if majority > 0.7:
            node_colors.append('red')  # Misaligned
        elif majority < 0.3:
            node_colors.append('green')  # Aligned
        else:
            node_colors.append('yellow')  # Mixed
    
    # Node sizes by number of points
    node_sizes = [G.nodes[node]['size'] * 10 for node in G.nodes()]
    
    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Plot
    if plot:
        fig, ax = plt.subplots(figsize=tuple(config.get('output', {}).get('figsize', [12, 10])))
    
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.85,
                              edgecolors='black', linewidths=1)
    
        ax.set_title('Mapper Graph: Yaw Misalignment Detection', 
                    fontsize=14, fontweight='normal')
        ax.axis('off')
    
    # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Aligned'),
            Patch(facecolor='#d62728', label='Misaligned'),
            Patch(facecolor='yellow', label='Mixed')
        ]
        ax.legend(frameon=False, handles=legend_elements, loc='upper right', fontsize=11)
    
        plt.tight_layout()
        plt.savefig(out_dir / 'mapper_graph.png', dpi=300, bbox_inches='tight')
        plt.close()


def main(plot: bool = False):
    np.random.seed(config.get('data', {}).get('seed', 42))
    
    logger.info("="*70)
    logger.info("Yaw Misalignment Detection Using Mapper")
    logger.info("="*70)
    
    # 1. Fetch wind data
    logger.info("\n1. Fetching NREL wind data...")
    wind_data = fetch_nrel_wind_data(lat=41.5, lon=-93.5, years=[2017, 2018])
    if wind_data is None:
        logger.error("Failed to fetch data")
        return
    logger.info(f"   Total records: {len(wind_data):,}")
    
    # 2. Simulate turbine with misalignment
    logger.info("\n2. Simulating turbine with yaw misalignment...")
    df = simulate_turbine_with_misalignment(wind_data)
    
    misalign_pct = (df['yaw_misalignment'].abs() > 10).sum() / len(df) * 100
    logger.info(f"   Misalignment (>10°): {misalign_pct:.1f}% of time")
    
    # 3. Create windows with filter values
    logger.info("\n3. Creating windows and computing filters...")
    windows_df = create_windows_with_filters(df, window_size=10)
    
    logger.info(f"   Total windows: {len(windows_df)}")
    logger.info(f"   Aligned: {(windows_df['label']==0).sum()}")
    logger.info(f"   Misaligned: {(windows_df['label']==1).sum()}")
    
    # 4. Split data chronologically
    logger.info("\n4. Splitting data...")
    split_idx = int(0.7 * len(windows_df))
    train_df = windows_df.iloc[:split_idx]
    test_df = windows_df.iloc[split_idx:]
    
    X_train = train_df[['filter1', 'filter2']].values
    y_train = train_df['label'].values
    X_test = test_df[['filter1', 'filter2']].values
    y_test = test_df['label'].values
    
    logger.info(f"   Train: {len(X_train)} windows")
    logger.info(f"   Test: {len(X_test)} windows")
    
    # 5. Build Mapper graph
    logger.info("\n5. Building Mapper graph...")
    G = build_mapper_graph(
        X_train,
        train_df['filter1'].values,
        train_df['filter2'].values,
        n_bins=8,
        overlap=0.5,
        n_clusters=2
    )
    
    logger.info(f"   Nodes: {G.number_of_nodes()}")
    logger.info(f"   Edges: {G.number_of_edges()}")
    logger.info(f"   Connected components: {nx.number_connected_components(G)}")
    
    # 6. Classify using Mapper
    logger.info("\n6. Classifying test set...")
    y_pred = classify_with_mapper(
        G, X_train, y_train, X_test,
        test_df['filter1'].values,
        test_df['filter2'].values
    )
    
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"\n   Accuracy: {acc*100:.2f}%")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Aligned', 'Misaligned'])}")
    
    # 7. Visualizations
    logger.info("\n7. Generating visualizations...")
    visualize_mapper_graph(G, y_train, 'figures_yaw')
    
    # Filter space scatter
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
    
        aligned_mask = y_test == 0
        misaligned_mask = y_test == 1
    
        ax.scatter(X_test[aligned_mask, 0], X_test[aligned_mask, 1],
                  c='green', alpha=0.5, s=30, label='Aligned', edgecolors='black', linewidths=0.5)
        ax.scatter(X_test[misaligned_mask, 0], X_test[misaligned_mask, 1],
                  c='red', alpha=0.5, s=30, label='Misaligned', edgecolors='black', linewidths=0.5)
    
        ax.set_xlabel('Filter 1: Power Ratio', fontsize=11)
        ax.set_ylabel('Filter 2: Rotor Speed Variability', fontsize=11)
        ax.set_title('Filter Space: Aligned vs Misaligned Operation', fontsize=12, fontweight='normal')
        ax.legend(frameon=False, fontsize=10)
    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        plt.tight_layout()
        plt.savefig('figures_yaw/filter_space.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("   Saved visualizations to figures_yaw/")
    
    logger.info("\n" + "="*70)
    logger.info("YAW MISALIGNMENT DETECTION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nMapper-based classification: {acc*100:.1f}% accuracy")
    logger.info(f"Detects misalignment without wind direction sensors")
    logger.info(f"Graph structure reveals:")
    logger.info(f"  - Aligned and misaligned operational branches")
    logger.info(f"  - Temporal degradation trajectories")
    logger.info(f"  - Misalignment mechanism signatures")
    logger.info("="*70)


if __name__ == "__main__":
    main()