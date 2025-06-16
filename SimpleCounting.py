import pandas as pd

def filter_high_process_counts(df, n):
    """
    Compute process counts from a DataFrame and return entries with counts > n.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns ['device', 'scan', 'procName'].
        n (int): Threshold for filtering counts.

    Returns:
        pd.DataFrame: Filtered DataFrame with process counts > n.
    """
    # Build process count matrix (device, scan, procName)
    scan_proc_counts = df.groupby(['device', 'scan', 'procName']).size().reset_index(name='count')

    # Filter only rows with count > n
    high_counts = scan_proc_counts[scan_proc_counts['count'] > n]

    # Optional: show all columns and enough rows for large outputs
    pd.set_option('display.max_rows', 150)
    pd.set_option('display.max_columns', None)

    # Display and return result
    print(high_counts)
    return high_counts
