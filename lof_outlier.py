###############################################################
################## LOF Outlier Detector #######################
###############################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

def detect_outlier_processes(df, z_thresh=3, count_thresh=50, contamination=0.01):
    """
    Detects anomalous processes in scan data using Local Outlier Factor.

    Parameters:
    - file_path (str): Path to the CSV file.
    - z_thresh (float): Z-score threshold for process anomaly detection.
    - count_thresh (int): Minimum count threshold for including a process.
    - contamination (float): LOF contamination parameter.

    Returns:
    - pd.DataFrame: Filtered DataFrame of anomalous processes with high count.
    """
    
    # Count processes per scan
    scan_level_df = (
        df.groupby(['device', 'scan', 'procName'])
          .size()
          .reset_index(name='count')
    )

    scan_vectors = scan_level_df.pivot_table(
        index=['device', 'scan'],
        columns='procName',
        values='count',
        fill_value=0
    ).reset_index()

    # Extract process columns
    proc_cols = scan_vectors.columns.difference(['device', 'scan'])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(scan_vectors[proc_cols])

    # LOF model
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_labels = lof.fit_predict(X_scaled)
    scan_vectors["lof_anomaly"] = lof_labels
    scan_vectors["lof_score"] = -lof.negative_outlier_factor_

    # Global stats
    mean_counts = scan_vectors[proc_cols].mean()
    std_counts = scan_vectors[proc_cols].std()

    anomalies = scan_vectors[scan_vectors["lof_anomaly"] == -1]

    # Collect high-z-score processes per anomalous scan
    records = []
    for _, row in anomalies.iterrows():
        device = row["device"]
        scan = row["scan"]
        lof_score = row["lof_score"]
        proc_values = row[proc_cols]

        z_scores = (proc_values - mean_counts) / (std_counts + 1e-6)
        high_z_procs = z_scores[z_scores > z_thresh]

        for proc in high_z_procs.index:
            count = proc_values[proc]
            if count > count_thresh:
                records.append({
                    "device": device,
                    "scan": scan,
                    "process": proc,
                    "count_per_scan": count,
                    "lof_score": lof_score
                })

    return pd.DataFrame(records)


