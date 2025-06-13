import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Load data
df = pd.read_csv("synthetic_iphone_latest.csv")

# Step 1: Build matrix of process counts per (device, scan)
scan_proc_counts = df.groupby(['device', 'scan', 'procName']).size().unstack(fill_value=0)

# Step 2: Save index info
index_info = scan_proc_counts.index.to_frame(index=False)

# Step 3: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(scan_proc_counts)

# Step 4: Run KMeans clustering
k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Step 5: Attach cluster labels to original index
scan_proc_counts['cluster'] = clusters
scan_proc_counts = scan_proc_counts.reset_index()

# Step 6: PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('KMeans Clustering of Device-Scan Process Profiles')
plt.colorbar(scatter, label='Cluster')
plt.show()

# Step 7: Show some results
for cluster_num in range(k):
    rows_in_cluster = scan_proc_counts[scan_proc_counts['cluster'] == cluster_num]
    print(f"\nCluster {cluster_num} has {len(rows_in_cluster)} (device, scan) entries")
    print(rows_in_cluster[['device', 'scan']].head())  # Show some samples


for cluster_num in scan_proc_counts['cluster'].unique():
    # Filter rows for this cluster
    cluster_rows = scan_proc_counts[scan_proc_counts['cluster'] == cluster_num]
    
    # Count how many times each device appears
    device_counts = cluster_rows['device'].value_counts()
    
    print(f"\nCluster {cluster_num} devices and counts (number of scans in cluster):")
    for device, count in device_counts.items():
        print(f"{device}: {count}")
        
feature_names = scan_proc_counts.columns.drop(['device', 'scan', 'cluster'])

centroids = kmeans.cluster_centers_

for i, centroid in enumerate(centroids):
    print(f"\nTop processes for Cluster {i}:")
    # Sort features by their centroid value descending
    top_indices = np.argsort(centroid)[::-1][:10]  # top 10 processes
    for idx in top_indices:
        print(f"{feature_names[idx]}: {centroid[idx]:.3f}")
