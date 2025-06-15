import pandas as pd
import os
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import ruptures as rpt

def analyze_df(df: pd.DataFrame, create_plots=False, output_dir='Statistics', anomaly_percentage=0.15):
    """
    Input:
    df a dataframe with columns: Device, scan, procName, scan_proc_count, timestamp.
    create_plots: Boolean :if True, generates plots for each device and saves them in the output directory.
    output_dir: Directory to save plots and statistics  
    anomaly_percentage:float between 0 and 1. Percentage of data to consider as anomalies.

    Output:
    Set of anomalous devices.
    Function returns a set of statistical anomalies detected in the data.
    Computation is based on mean value and standard deviation of the difference between 
    the total process count and the scan process count for each device.
    The outliers are detected using Isolation Forest algorithm.
    """

    # Create output directory if needed
    if create_plots:
        os.makedirs(output_dir, exist_ok=True)

    # --- Preprocessing ---
    df['total_proc_count'] = df.groupby(['device', 'scan'])['procName'].transform('count')
    # Computing scan_proc_count as the count of unique procNames per device and scan
    df['difference_count'] = df['total_proc_count'] - df['scan_proc_count']


    # --- Prepare data without procNames ---
    data_no_proc = df.drop(columns=["Unnamed: 0", "procName", 'scan_proc_count', 'timestamp', 'total_proc_count']).drop_duplicates()
    #Compute mean and std of difference_count per device
    data_no_proc['mean_difference_count'] = data_no_proc.groupby('device')['difference_count'].transform('mean')
    data_no_proc['std_difference_count'] = data_no_proc.groupby('device')['difference_count'].transform('std')

    # --- Optional: Generate Plots per Device ---
    if create_plots:
        for device_name in sorted(data_no_proc['device'].unique()):
            result = data_no_proc[data_no_proc['device'] == device_name]
            if result.empty:
                continue

            result['difference_count'] = pd.to_numeric(result['difference_count'])
            y_axis_col = 'difference_count'
            mean_value = result[y_axis_col].mean()

            plt.figure(figsize=(10, 6))
            plt.plot(result.index, result[y_axis_col], marker='o', label='Difference Count')
            plt.axhline(y=mean_value, color='red', linestyle='-.', label=f'Mean = {mean_value:.2f}')

            signal = result[y_axis_col].values.reshape(-1, 1)
            algo = rpt.Pelt(model="l2").fit(signal)
            change_points = algo.predict(pen=1000)

            for cp in change_points[:-1]:
                plt.axvline(x=cp, color='blue', linestyle='--', alpha=0.6,
                            label='Change Point' if cp == change_points[0] else None)

            plt.xticks(ticks=result.index, labels=result['scan'], rotation=45, ha='right')
            plt.title(f'Difference Count for {device_name}')
            plt.xlabel('Scan Number')
            plt.ylabel('Difference Count')
            plt.grid(True)
            plt.tight_layout()
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            filename = os.path.join(output_dir, f'{device_name}_statistics.png')
            plt.savefig(filename)
            plt.close()

            change_labels = [result.iloc[cp].scan for cp in change_points[:-1]]
            #print(f"🔍 Change points for {device_name}: {change_labels}")

    # --- Anomaly Detection ---
    filtered_df = data_no_proc.drop(columns=['scan', 'difference_count']).drop_duplicates()
    features = filtered_df[['mean_difference_count', 'std_difference_count']]
    iso_forest = IsolationForest(contamination=anomaly_percentage, random_state=42)
    filtered_df['anomaly'] = iso_forest.fit_predict(features)
    filtered_df['anomaly_score'] = iso_forest.decision_function(features)

    scaler = MinMaxScaler()
    filtered_df['score_normalized'] = scaler.fit_transform(-filtered_df[['anomaly_score']])  # negate to score higher = more anomalous

    for idx, row in filtered_df.iterrows():
        device = row['device']
        score = row['anomaly_score']
        if row['anomaly'] == -1:
            print(f"Attention: Device {device} is an anomaly (score = {score:.4f}) — needs investigation.")
        #else:
            #print(f"Device {device} appears normal (score = {score:.4f}).")

    anomalous_devices = set(filtered_df.loc[filtered_df['anomaly'] == -1, 'device'])
    return anomalous_devices


# df = pd.read_csv("synthetic_iphone_new_2.csv")
# anomalous_devices = analyze_df(df, output_dir='Statistics', create_plots=False, anomaly_percentage=0.15)
# print(f"Anomalous devices detected: {anomalous_devices}")