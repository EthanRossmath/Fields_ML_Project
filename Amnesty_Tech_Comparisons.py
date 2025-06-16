from rapidfuzz import fuzz, process
import pandas as pd

def fuzzy_check(name, known_list, threshold):
    """
    Perform fuzzy match using token_set_ratio. Return best match if above threshold and length ratio is acceptable.
    """
    name_lower = name.lower()
    match, score, _ = process.extractOne(name_lower, known_list, scorer=fuzz.token_set_ratio)
    
    if score >= threshold:
        len_ratio = min(len(name_lower), len(match)) / max(len(name_lower), len(match))
        if len_ratio >= 0.7:
            return match, score

    return None, 0

def detect_fuzzy_matched_processes(df, known_processes_file, threshold=100):
    """
    Identify suspicious processes in a DataFrame by fuzzy matching against a known list.

    Parameters:
        df (pd.DataFrame): Must contain a 'procName' and 'device' column.
        known_processes_file (str): Path to text file of known malicious process names (one per line).
        threshold (int): Minimum fuzz match score (0-100) to accept.

    Returns:
        List[Tuple[str, str, int]]: Matched processes as (original, matched, score).
        Dict[str, List[str]]: Mapping from process name to list of devices that used it.
    """
    # Get distinct process names
    distinct_procs = df['procName'].dropna().unique().tolist()

    # Load known process names from file
    with open(known_processes_file, 'r', encoding='utf-8') as f:
        known_procs = [line.strip().lower() for line in f if line.strip()]

    proc_matches = []
    device_map = {}

    # Match each process against known list
    for proc in distinct_procs:
        match, score = fuzzy_check(proc, known_procs, threshold)
        if match:
            proc_matches.append((proc, match, score))
            matching_devices = df[df['procName'] == proc]['device'].unique().tolist()
            device_map[proc] = matching_devices

    return proc_matches, device_map

def print_matched_processes(proc_matches):
    """
    Print fuzzy match results.
    """
    print("\n=== Fuzzy Matches in Known Malicious Processes ===")
    for orig, matched, score in proc_matches:
        print(f"{orig}  -->  {matched}  (score: {score})")

def print_affected_devices(proc_matches, device_map):
    """
    Print devices affected by matched malicious processes.
    """
    print("\n=== Devices with Matched Malicious Processes ===")
    for orig_proc, matched_proc, score in proc_matches:
        devices = device_map.get(orig_proc, [])
        device_str = ", ".join(devices) if devices else "No devices found"
        print(f"Process: {orig_proc}  --> Matched: {matched_proc}  (score: {score})")
        print(f"Devices: {device_str}\n")
