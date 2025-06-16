import pandas as pd
from Kmeans_clustering import get_devices_not_in_top_clusters
from lof_outlier import lof_outliers
from statistical_analysis import analyze_df
from fuzzy_search import detect_anomalous_devices
from DBSCAN import anomalous_devices_DBSCAN
from SimpleCounting import filter_high_process_counts
from Amnesty_Tech_Comparisons import detect_fuzzy_matched_processes

###############################################
df = pd.read_csv("synthetic_iphone_latest.csv")
###############################################

# The list of known malicious processes involved in the Pegasus project 
# made publicly available by Amnesty Tech.
known_proc_file = "/home/seed/all_processes.txt"

# Compare the processes in our dataframe to the known list of Malicious processes (The threshold is the threshold of the fuzzysearch algorithm)
proc_matches, device_map = detect_fuzzy_matched_processes(df, known_proc_file, threshold=100)

# Counts the number of times a process appears in a (device,scan) pair and lists the processes and the (device,scan) 
# pair for which it had a count more than $n$. For this data, $n=100$ is more or less optimal.
#Counting_simple = filter_high_process_counts(df,100)

# Does the Kmeans clustering (10 clusters) and outputs the devices that are not in the top 2 clusters.
kmeans_malicious = get_devices_not_in_top_clusters(df)

# Does the DBSCAN clustering and outputs the noisy devices.
DBSCAN_malicious = anomalous_devices_DBSCAN(df)

# Returns a dataframe consisting of devices, scans, process names, and the number of times a process apppears
# in that scan. Based on Local Outlier Factor algorithm to find outliers in a dataset. 
lof = lof_outliers(df, count_thresh=0, contamination=0.01)
lof_device = set(lof['device'].unique())
lof_device = set(lof['device'].unique())
lof_process_df = lof.loc[lof['count_per_scan']>30]
lof_process = set(lof_process_df['process'])

#Returns a set of anomaly_percentage Devices, based on statistical analysis of procedure count (total - unique)
#Set create_plots to True, to save graphs for each device in the output_dir.
stat = analyze_df(df, create_plots=False, output_dir='Statistics',  anomaly_percentage=0.15, verbose=False)

#Returns a set of n devices that are suspicious depending on a fuzzy search of procName,
#i.e. if the process name has uncommon special characters or is similar to another procName
#but not exactly the same. Set verbose to True to see a list of procNames together with their devices.
fuzzy_search = detect_anomalous_devices(df, n=8, verbose=False)


# Printing the results of the analysis.
def print_unusual_behaviour(set0, set1, set2, set3, set4):
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    # formatting the outputs so only 7 items appear per line (for readability)
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
        f"{BOLD}{UNDERLINE}Devices that have an unusual number of crashing processes KMeans:{END}\n"
        f"{format_set_with_line_breaks(set0)}\n\n"
        f"{BOLD}{UNDERLINE}Devices that have an unusual number of crashing processes LOF:{END}\n"
        f"{format_set_with_line_breaks(set1)}\n\n"
        f"{BOLD}{UNDERLINE}Processes that crash an unusual number of times:{END}\n"
        f"{format_set_with_line_breaks(set2)}\n\n"
        f"{BOLD}{UNDERLINE}Devices that have unusual statistical process count behaviour:{END}\n"
        f"{format_set_with_line_breaks(set3)}\n\n"
        f"{BOLD}{UNDERLINE}Devices which have strange letter combinations for processes:{END}\n"
        f"{format_set_with_line_breaks(set4)}\n"
    )
    print(output)


print_unusual_behaviour(kmeans_malicious, lof_device, lof_process, stat, fuzzy_search)


