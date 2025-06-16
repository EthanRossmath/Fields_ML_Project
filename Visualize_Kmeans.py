import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def visualize_kmeans_clusters(df, n_clusters=10):
    # Step 1: Create (device, scan) x procName matrix
    scan_proc_counts = df.groupby(['device', 'scan', 'procName']).size().unstack(fill_value=0)

    # Step 2: Scale features
    X_scaled = StandardScaler().fit_transform(scan_proc_counts)

    # Step 3: Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Step 4: Reduce dimensions to 2D with PCA for visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    # Step 5: Plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(components[:, 0], components[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    plt.title(f"KMeans Clusters Visualization (k={n_clusters})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label='Cluster Label')
    plt.show()
