import pandas as pd
from rapidfuzz import fuzz
import jellyfish
from collections import defaultdict

def detect_anomalous_devices(df: pd.DataFrame, n: int = 10, threshold: int = 85, verbose: bool = False) -> set:
    """
    Detects anomalous devices based on rare and suspicious process names.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with 'procName' and 'device' columns.
        n (int): Number of top anomalous devices to return.
        threshold (int): Similarity threshold for fuzzy matching (default = 85).
        verbose (bool): If True, prints each procName with its associated devices.

    Returns:
        Set[str]: Top n anomalous devices.
    """
    
    # Step 1: Group similar procNames
    unique_proc_names = df['procName'].dropna().unique()
    groups = []
    procname_to_rep = {}

    for name in unique_proc_names:
        sound_code = jellyfish.soundex(name)
        found_group = False

        for group in groups:
            rep = group[0]
            rep_sound = jellyfish.soundex(rep)
            if sound_code == rep_sound and fuzz.ratio(name, rep) >= threshold:
                group.append(name)
                procname_to_rep[name] = rep
                found_group = True
                break

        if not found_group:
            groups.append([name])
            procname_to_rep[name] = name

    # Keep only groups with more than one member
    meaningful_groups = [group for group in groups if len(group) > 1]

    # Step 2: Build procName → set of devices
    procname_to_devices = {}
    for group in meaningful_groups:
        for name in group:
            devices = set(df.loc[df['procName'] == name, 'device'].dropna())
            procname_to_devices[name] = devices

    # Optional print
    if verbose:
        for name, devices in procname_to_devices.items():
            print(f"{name}:")
            print(f"  ➤ Devices: {sorted(devices)}\n")

    # Step 3: Score devices based on rarity and suspicious name pattern
    device_scores = defaultdict(float)

    for group in meaningful_groups:
        proc_counts = {name: len(procname_to_devices[name]) for name in group}
        max_count = max(proc_counts.values()) or 1

        for name in group:
            devices = procname_to_devices[name]
            count = proc_counts[name]

            # Anomaly scoring
            rarity_score = 1 - (count / max_count)
            suspicious_name = int(name.strip().lower().endswith(('_', '*', '!', '-', '+')))
            total_score = rarity_score + 0.5 * suspicious_name

            for device in devices:
                device_scores[device] += total_score

    # Step 4: Sort and return top n anomalous devices
    ranked_anomalous_devices = sorted(device_scores.items(), key=lambda x: x[1], reverse=True)
    top_devices = {device for device, _ in ranked_anomalous_devices[:n]}
    
    return top_devices


# df = pd.read_csv("synthetic_iphone_new_2.csv")
# top_anomalous = detect_anomalous_devices(df, n=8, verbose=True)
# print(top_anomalous)