import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from kneed import KneeLocator

def build_scan_proc_counts(df):
    """
    Create a pivot table with (device, scan) as rows and process names as columns.
    Values are counts of each process in a scan.
    """
    return df.groupby(['device', 'scan', 'procName']).size().unstack(fill_value=0)

def scale_features(df_counts):
    """
    Standardize the features to zero mean and unit variance.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df_counts)

def reduce_dimensionality(X, n_components=10):
    """
    Apply PCA to reduce dimensionality of the dataset.
    """
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)

def compute_k_distances(X, k=5):
    """
    Compute the sorted distances to each point's k-th nearest neighbor.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    return np.sort(distances[:, k-1])

def find_elbow_point(distances):
    """
    Automatically detect the elbow in the k-distance curve using KneeLocator.
    Falls back to median if no elbow is detected.
    """
    kl = KneeLocator(
        x=range(len(distances)),
        y=distances,
        curve='convex',
        direction='increasing'
    )
    elbow_index = kl.knee
    if elbow_index is None:
        return np.median(distances)
    return distances[elbow_index]

def run_dbscan(X, eps, min_samples=5):
    """
    Run the DBSCAN clustering algorithm.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(X)

def anomalous_devices_DBSCAN(df, pca_components=10, k=5):
    """
    Detect anomalous devices using DBSCAN clustering.

    Parameters:
        df (pd.DataFrame): Must contain columns 'device', 'scan', 'procName'.
        pca_components (int): Number of PCA components to retain.
        k (int): Neighbors to use in k-distance for eps selection.

    Returns:
        List[str]: Unique device IDs labeled as noise by DBSCAN.
    """
    scan_proc_counts = build_scan_proc_counts(df)
    X_scaled = scale_features(scan_proc_counts)
    X_reduced = reduce_dimensionality(X_scaled, n_components=pca_components)
    kth_distances = compute_k_distances(X_reduced, k=k)
    eps = find_elbow_point(kth_distances)

    # Optional: clamp very large eps values
    eps = min(eps, 5.0)

    clusters = run_dbscan(X_reduced, eps=eps, min_samples=5)
    scan_proc_counts = scan_proc_counts.reset_index()
    scan_proc_counts['cluster'] = clusters

    noise_points = scan_proc_counts[scan_proc_counts['cluster'] == -1]
    return noise_points['device'].unique().tolist()
