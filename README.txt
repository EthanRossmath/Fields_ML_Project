Python packages needed to run.

    pandas, numpy, sklearn.ensemble, sklearn.preprocessing, sklearn.neighbors, sklearn.cluster,
    sklearn.decomposition, kneed, matplotlib.pyplot, rapidfuzz, jellyfish, collections, os, ruptures

Steps to run the code.

    1. Upload .csv file containing iphone scans to the directory containing Main.py

    2. Ensure the .csv file is formatted so that the device names have column title "device",
    scan IDs have column title "scan", process names have title "procName", and the counts of
    process running in each scan has column title "scan_proc_count".

    3. Go to Main.py and on line 12, replace synthetic_iphone_latest.csv with the name of you .csv
    file.

    4. Run the file and have fun :)


Special Notes:
    The statistical analysis can output pictures if one sets create_plots=True. The process will take a couple
    of minutes with this feature switched on. The pictures for the standard data are also offered in the statistics folder
    stat = analyze_df(df, create_plots=False, output_dir='Statistics',  anomaly_percentage=0.15, verbose=False)

