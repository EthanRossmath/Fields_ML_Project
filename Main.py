import pandas as pd
from KMEANSNEW import build_scan_proc_counts
from lof_outlier import detect_outlier_processes
from statistical_analysis import analyze_df
from fuzzy_search import detect_anomalous_devices
df = pd.read_csv("synthetic_iphone_latest.csv")

print(build_scan_proc_counts(df))

detect_outlier_processes(df)

#Returns a set of anomaly_percentage Devices, based on statistical analysis of procedure count (total - unique)
#Set create_plots to True, to save graphs for each device in the output_dir.
analyze_df(df, create_plots=False, output_dir='Statistics',  anomaly_percentage=0.15)

#Returns a set of n devices that are suspicious depending on a fuzzy search of procName,
#i.e. if the process name has uncommon special characters or is similar to another procName
#but not exactly the same. Set verbose to True to see a list of procNames together with their devices.
detect_anomalous_devices(df, n=8, verbose=False)
