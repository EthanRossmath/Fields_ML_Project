from rapidfuzz import fuzz, process
import pandas as pd

# Inline known process names
KNOWN_PROCESSES = [
    'ABSCarryLog', 'accountpfd', 'actmanaged', 'aggregatenotd', 'appccntd', 'bfrgbd',
    'bh', 'bluetoothfs', 'boardframed', 'brstaged', 'brfstagingd', 'bundpwrd',
    'cfprefssd', 'ckeblld', 'ckkeyrollfd', 'com.apple.Mappit.SnapshotService',
    'com.apple.rapports.events', 'CommsCenterRootHelper', 'comnetd', 'comsercvd',
    'confinstalld', 'contextstoremgrd', 'corecomnetd', 'ctrlfs', 'dhcp4d',
    'Diagnostic-2543', 'Diagnosticd', 'Diagnostics-2543', 'eventfssd', 'eventsfssd',
    'eventstorpd', 'faskeepd', 'fdlibframed', 'fmld', 'frtipd', 'fservernetd',
    'gatekeeperd', 'GoldenGate', 'gssdp', 'JarvisPluginMgr', 'jlmvskrd', 'launchafd',
    'launchrexd', 'libbmanaged', 'libtouchregd', 'llmdwatchd', 'lobbrogd', 'locserviced',
    'logseld', 'misbrigd', 'mobileargd', 'MobileSMSd', 'mptbd', 'msgacntd', 'natgd',
    'neagentd', 'nehelprd', 'netservcomd', 'otpgrefd', 'passsd', 'payload', 'pcsd',
    'PDPDialogs', 'pstid', 'ReminderIntentsUIExtension', 'rlaccountd', 'roleaboutd',
    'roleaccountd', 'rolexd', 'seraccountd', 'setframed', 'smmsgingd', 'stagegrad',
    'stagingd', 'vm_stats', 'keybrd', 'xpccfd', 'fnotifyd', 'tisppd', 'updaterd',
    'wifip2ppd'
]

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

def detect_fuzzy_matched_processes(df, known_processes=KNOWN_PROCESSES, threshold=100):
    """
    Identify suspicious processes in a DataFrame by fuzzy matching against a known list.

    Parameters:
        df (pd.DataFrame): Must contain 'procName' and 'device' columns.
        known_processes (List[str]): List of known suspicious process names.
        threshold (int): Minimum fuzzy match score (0-100) to consider a match.

    Returns:
        List[Tuple[str, str, int]]: Matched processes as (original, matched, score).
        Dict[str, List[str]]: Mapping from original process name to devices that used it.
    """
    known_lower = [p.lower() for p in known_processes]
    distinct_procs = df['procName'].dropna().unique().tolist()

    proc_matches = []
    device_map = {}

    for proc in distinct_procs:
        match, score = fuzzy_check(proc, known_lower, threshold)
        if match:
            proc_matches.append((proc, match, score))
            matching_devices = df[df['procName'] == proc]['device'].unique().tolist()
            device_map[proc] = matching_devices

    return proc_matches, device_map
