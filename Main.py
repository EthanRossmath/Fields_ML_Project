import pandas as pd
from KMEANSNEW import build_scan_proc_counts
from lof_outlier import detect_outlier_processes
from statistical_analysis import analyze_df
df = pd.read_csv("synthetic_iphone_latest.csv")

print(build_scan_proc_counts(df))

detect_outlier_processes(df)

#Returns a set of anomaly_percentage Devices, based on statistical analysis of procedure count (total - unique)
#Set create_plots to True, to save graphs for each device in the output_dir.
analyze_df(df, create_plots=False, output_dir='Statistics',  anomaly_percentage=0.15)
