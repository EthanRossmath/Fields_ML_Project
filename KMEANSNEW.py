import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def build_scan_proc_counts(df):
    """Return matrix: one row per (device, scan), columns are process names (counts)."""
    scan_proc_counts = df.groupby(['device', 'scan', 'procName']).size().unstack(fill_value=0)
    return scan_proc_counts


def scale_features(scan_proc_counts):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(scan_proc_counts)
    return X_scaled, scaler


def perform_kmeans(X_scaled, n_clusters=10):
    """Run KMeans clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters


def add_clusters_to_data(scan_proc_counts, clusters):
    """Attach cluster labels and reset index."""
    scan_proc_counts = scan_proc_counts.copy()
    scan_proc_counts['cluster'] = clusters
    scan_proc_counts = scan_proc_counts.reset_index()
    return scan_proc_counts


def plot_pca(X_scaled, clusters):
    """Visualize clusters using PCA."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('KMeans Clustering of Device-Scan Process Profiles')
    plt.colorbar(scatter, label='Cluster')
    plt.show()


def print_cluster_summaries(scan_proc_counts, k):
    for cluster_num in range(k):
        rows_in_cluster = scan_proc_counts[scan_proc_counts['cluster'] == cluster_num]
        print(f"\nCluster {cluster_num} has {len(rows_in_cluster)} (device, scan) entries")
        print(rows_in_cluster[['device', 'scan']].head())


def print_device_counts_per_cluster(scan_proc_counts):
    for cluster_num in scan_proc_counts['cluster'].unique():
        cluster_rows = scan_proc_counts[scan_proc_counts['cluster'] == cluster_num]
        device_counts = cluster_rows['device'].value_counts()
        print(f"\nCluster {cluster_num} devices and counts (number of scans in cluster):")
        for device, count in device_counts.items():
            print(f"{device}: {count}")


def print_top_processes_per_cluster(kmeans, feature_names, top_n=10):
    centroids = kmeans.cluster_centers_

    for i, centroid in enumerate(centroids):
        print(f"\nTop {top_n} processes for Cluster {i}:")
        top_indices = np.argsort(centroid)[::-1][:top_n]
        for idx in top_indices:
            print(f"{feature_names[idx]}: {centroid[idx]:.3f}")


def full_pipeline(df, n_clusters=10):
    scan_proc_counts = build_scan_proc_counts(df)
    X_scaled, scaler = scale_features(scan_proc_counts)
    kmeans, clusters = perform_kmeans(X_scaled, n_clusters)
    scan_proc_counts_with_clusters = add_clusters_to_data(scan_proc_counts, clusters)

    plot_pca(X_scaled, clusters)
    print_cluster_summaries(scan_proc_counts_with_clusters, n_clusters)
    print_device_counts_per_cluster(scan_proc_counts_with_clusters)

    feature_names = scan_proc_counts.columns.tolist()
    print_top_processes_per_cluster(kmeans, feature_names)

    return scan_proc_counts_with_clusters, kmeans
