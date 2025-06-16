import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from kneed import KneeLocator
import matplotlib.pyplot as plt

def build_scan_proc_counts(df):
    """
    Create a pivot table with (device, scan) as rows and process names as columns.
    Values are counts of each process in a scan.
    """
    return df.groupby(['device', 'scan', 'procName']).size().unstack(fill_value=0)

def scale_features(df_counts):
    """
    Standardize the features to zero mean and unit variance.
    Required for PCA and distance-based methods like DBSCAN.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df_counts)

def reduce_dimensionality(X, n_components=10):
    """
    Apply PCA to reduce dimensionality of the dataset.
    Reduces noise and speeds up clustering.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    return X_reduced

def plot_k_distance(X, k=5):
    """
    Plot the sorted distances to each point's k-th nearest neighbor.
    Used to identify the optimal 'eps' value for DBSCAN.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    kth_distances = np.sort(distances[:, k-1])

    plt.figure(figsize=(8, 5))
    plt.plot(kth_distances)
    plt.title(f'k-distance Graph (k={k})')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}th Nearest Neighbor Distance')
    plt.grid(True)
    plt.show()

    return kth_distances

def find_elbow_point(distances):
    """
    Automatically detect the elbow in the k-distance curve using KneeLocator.
    This gives a good eps value for DBSCAN.
    """
    kl = KneeLocator(
        x=range(len(distances)),
        y=distances,
        curve='convex',
        direction='increasing'
    )
    elbow_index = kl.knee

    if elbow_index is None:
        # Fall back to median if no elbow found
        print("Warning: No elbow detected. Using median distance.")
        return np.median(distances)

    return distances[elbow_index]

def run_dbscan(X, eps, min_samples=5):
    """
    Run the DBSCAN clustering algorithm.
    Points labeled -1 are considered noise (outliers).
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(X)

def anomalous_devices_DBSCAN(df, pca_components=10, k=5, show_plot=True):
    """
    Full pipeline to detect anomalous (outlier) devices using DBSCAN.

    Parameters:
        df (pd.DataFrame): Must contain columns 'device', 'scan', 'procName'.
        pca_components (int): Number of PCA components to retain.
        k (int): Neighbors to use in k-distance for eps selection.
        show_plot (bool): Whether to show the k-distance plot.

    Returns:
        List[str]: Unique device IDs that were identified as noise.
    """
    # Step 1: Build (device, scan) x process count matrix
    scan_proc_counts = build_scan_proc_counts(df)

    # Step 2: Standardize the features
    X_scaled = scale_features(scan_proc_counts)

    # Step 3: Reduce dimensionality via PCA
    X_reduced = reduce_dimensionality(X_scaled, n_components=pca_components)

    # Step 4: Plot and analyze k-distance graph for eps
    if show_plot:
        kth_distances = plot_k_distance(X_reduced, k=k)
    else:
        # Skip plotting, just calculate distances
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_reduced)
        kth_distances = np.sort(nbrs.kneighbors(X_reduced)[0][:, k-1])

    # Step 5: Find best eps using elbow method
    eps = find_elbow_point(kth_distances)
    print(f"Chosen eps: {eps:.4f}")

    # Optional sanity check: clip large eps values
    if eps > 10:
        print("Warning: Eps is very large; adjusting to 5.0.")
        eps = 5.0

    # Step 6: Run DBSCAN clustering
    clusters = run_dbscan(X_reduced, eps=eps, min_samples=5)

    # Step 7: Add cluster labels back to the table
    scan_proc_counts = scan_proc_counts.reset_index()
    scan_proc_counts['cluster'] = clusters

    # Step 8: Extract all devices that were labeled as noise (-1)
    noise_points = scan_proc_counts[scan_proc_counts['cluster'] == -1]
    noise_devices = noise_points['device'].unique().tolist()

    return noise_devices
