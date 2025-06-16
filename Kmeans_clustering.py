from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_devices_not_in_top_clusters(df, n_clusters=10, top_n=2):
    """
    Returns devices that are NOT in the top_n largest clusters based on (device, scan) profiles.

    Parameters:
        df (DataFrame): Input dataframe containing 'device', 'scan', and 'procName' columns
        n_clusters (int): Number of clusters to form with KMeans
        top_n (int): Number of largest clusters to exclude (considered "normal")

    Returns:
        Set of device IDs that belong to clusters other than the top_n largest clusters
    """

    # Step 1: Aggregate data to get count of each process per (device, scan)
    # Creates a matrix: rows = (device, scan), columns = processes, values = counts
    scan_proc_counts = df.groupby(['device', 'scan', 'procName']).size().unstack(fill_value=0)

    # Step 2: Standardize the feature matrix for better clustering performance
    X_scaled = StandardScaler().fit_transform(scan_proc_counts)

    # Step 3: Perform KMeans clustering on scaled data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Step 4: Add cluster assignments back to the aggregated dataframe
    scan_proc_counts['cluster'] = clusters
    scan_proc_counts = scan_proc_counts.reset_index()

    # Step 5: Find the sizes of each cluster (number of (device, scan) entries per cluster)
    cluster_sizes = scan_proc_counts['cluster'].value_counts()

    # Step 6: Identify the top_n largest clusters by size (these are considered "normal" clusters)
    top_clusters = cluster_sizes.nlargest(top_n).index.tolist()

    # Step 7: Filter rows belonging to clusters NOT in the top_n largest clusters
    non_top_rows = scan_proc_counts[~scan_proc_counts['cluster'].isin(top_clusters)]

    # Step 8: Extract unique devices that belong to these smaller/anomalous clusters
    anomalous_devices = set(non_top_rows['device'])

    return anomalous_devices
