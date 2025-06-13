import pandas as pd
from KMEANSNEW import scan_proc_counts
from lof_outlier import detect_outlier_processes


df = pd.read_csv("synthetic_iphone_latest.csv")

print(scan_proc_counts)
