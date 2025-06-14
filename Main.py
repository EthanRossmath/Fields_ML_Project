import pandas as pd
from KMEANSNEW import build_scan_proc_counts
from lof_outlier import lof_outliers
from statistical_analysis import analyze_df
from fuzzy_search import detect_anomalous_devices
df = pd.read_csv("synthetic_iphone_latest.csv")

kmeans = build_scan_proc_counts(df)

# Returns a dataframe consisting of devices, scans, process names, and the number of times a process apppears
# in that scan. Based on Local Outlier Factor algorithm to find outliers in a dataset. 
lof = lof_outliers(df, count_thresh=50, contamination=0.01)
lof_device = set(lof['device'].unique())
lof_process = set(lof['process'].unique())

#Returns a set of anomaly_percentage Devices, based on statistical analysis of procedure count (total - unique)
#Set create_plots to True, to save graphs for each device in the output_dir.
stat = analyze_df(df, create_plots=False, output_dir='Statistics',  anomaly_percentage=0.15)

#Returns a set of n devices that are suspicious depending on a fuzzy search of procName,
#i.e. if the process name has uncommon special characters or is similar to another procName
#but not exactly the same. Set verbose to True to see a list of procNames together with their devices.
fuzzy_search = detect_anomalous_devices(df, n=8, verbose=False)


#### ETHAN HERE: JUST MESSING AROUND, LET ME KNOW WHAT YOU THINK ############

def print_unusual_behaviour(set1, set2, set3, set4):
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    def format_set_with_line_breaks(s, max_items_per_line=7):
        if not s:
            return "(None)"
        items = sorted(str(item) for item in s)
        lines = [
            ", ".join(items[i:i+max_items_per_line])
            for i in range(0, len(items), max_items_per_line)
        ]
        return "\n".join(lines)

    output = (
        f"{BOLD}{UNDERLINE}Devices that have an unusual number of crashing processes:{END}\n"
        f"{format_set_with_line_breaks(set1)}\n\n"
        f"{BOLD}{UNDERLINE}Processes that crash an unusual number of times:{END}\n"
        f"{format_set_with_line_breaks(set2)}\n\n"
        f"{BOLD}{UNDERLINE}Devices that have unusual statistical process count behaviour:{END}\n"
        f"{format_set_with_line_breaks(set3)}\n\n"
        f"{BOLD}{UNDERLINE}Devices which have strange letter combinations for processes:{END}\n"
        f"{format_set_with_line_breaks(set4)}\n"
    )
    print(output)


print_unusual_behaviour(lof_device, lof_process, stat, fuzzy_search)


