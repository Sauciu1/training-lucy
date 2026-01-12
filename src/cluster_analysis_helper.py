from math import log
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor, load_results


def get_standing_clip_test_folders(base_dir="cluster_copy", prefix="standing_clip_test_"):
    """Return all folder names in base_dir that start with prefix."""
    return [f for f in os.listdir(base_dir) if f.startswith(prefix) and os.path.isdir(os.path.join(base_dir, f))]

def collect_log_and_info_files(base_dir="cluster_copy", prefix="standing_clip_test_"):
    """Return lists of episode_log.csv and run_info.json file paths from relevant folders."""
    folders = get_standing_clip_test_folders(base_dir, prefix)
    log_files, info_files = [], []
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        log_path = os.path.join(folder_path, "episode_log.csv")
        info_path = os.path.join(folder_path, "run_info.json")
        if os.path.exists(log_path):
            log_files.append(log_path)
        else:
            monitor_log = get_folder_child_folder_with_file(folder_path, "0.monitor.csv")
            log_files.append(monitor_log)
        if os.path.exists(info_path):
            info_files.append(info_path)


    return log_files, info_files

def get_folder_child_folder_with_file(folder, filename):
    """Return the first child folder of folder that contains filename."""
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            target_file = os.path.join(item_path, filename)
            if os.path.exists(target_file):
                return item_path
    return None

def read_and_combine_logs(log_files):
    """Read and combine episode_log.csv files into a single DataFrame."""
    logs = []
    for log_file in log_files:
        
    

        if log_file.endswith(".csv"):
            df = pd.read_csv(log_file)
            df['source_folder'] = os.path.basename(os.path.dirname(log_file))
            logs.append(df)
            return pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
        else:
            df = load_results(log_file)
            return df

def read_infos_as_dict(info_files):
    """Read run_info.json files and return a list of dicts with source_folder info."""
    infos = []
    for info_file in info_files:
        with open(info_file, 'r') as f:
            info = json.load(f)
            info['source_folder'] = os.path.basename(os.path.dirname(info_file))
            infos.append(info)
    return infos


def find_info_dict(folder, infos_dicts):
    """Find the matching info dict for a folder from one or more infos_dict lists."""
    if not isinstance(infos_dicts, list):
        infos_dicts = [infos_dicts]
    for infos in infos_dicts:
        match = next((info for info in infos if info.get('source_folder') == folder), None)
        if match is not None:
            return match
    return None

def recursive_search_dict(d, key):
    """Recursively search for a key in a dict or list."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            result = recursive_search_dict(v, key)
            if result is not None:
                return result
    elif isinstance(d, list):
        for item in d:
            result = recursive_search_dict(item, key)
            if result is not None:
                return result
    return None

def add_param_to_logs_df(logs_df, infos_dicts, param_name):
    """For each row in logs_df, get the param_name from the matching info dict(s) and add it as a new column."""
    param_values = []
    for _, row in logs_df.iterrows():
        folder = row['source_folder']
        match = find_info_dict(folder, infos_dicts)
        value = None
        if match is not None:
            value = recursive_search_dict(match, param_name)
        param_values.append(value)
    logs_df[param_name] = param_values
    return logs_df


def rolling_average_lineplot(data, x_col, y_col, timestep_average=1000, color=None, label=None, **kwargs):
    
    ax = plt.gca()
    df = data.sort_values(by=x_col)
    # Rolling average (red)
    df["rolling_avg"] = df[y_col].rolling(window=timestep_average, min_periods=1).mean()
    ax.plot(
        df[x_col],
        df["rolling_avg"],
        color="red",
        label=f"Rolling Average of {y_col} (window={timestep_average})",
        **kwargs,
    )
    # Actual average (light blue, alpha=0.3)
    ax.plot(df[x_col], df[y_col], color="skyblue", alpha=0.4, label=f"Actual {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True)
    #ax.set_title(f"Rolling Average Line Plot of {y_col} vs {x_col}")
    ax.legend()



def rolling_hue_lineplot(
    df, x_col, y_col, hue=None, n_values=100,
    ax=None, cmap="viridis", alpha=0.6, figsize=(8, 6), hue_order=None
):
    """
    Plot rolling mean over a fixed number of values (n_values) rather than window size in x_col.
    """
    ax = ax or plt.figure(figsize=figsize).gca()
    cols = [x_col, y_col] + ([hue] if hue and hue in df.columns else [])
    d = df[cols].dropna().sort_values(x_col)

    def rolling_mean_by_count(s, n):
        return s.rolling(window=n, min_periods=1).mean()

    if hue and hue in d.columns:
        is_numeric_hue = pd.api.types.is_numeric_dtype(d[hue])
        groups = list(d.groupby(hue, sort=is_numeric_hue))
        if is_numeric_hue:
            groups = sorted(groups, key=lambda kv: kv[0])
        if hue_order is not None:
            groups = sorted(groups, key=lambda kv: hue_order.index(kv[0]) if kv[0] in hue_order else -1)

        colors = cm.get_cmap(cmap)(
            np.linspace(0.15, 0.95, len(groups))
        )

        for (name, g), c in zip(groups, colors):
            g = g.sort_values(x_col)
            rm = rolling_mean_by_count(g[y_col], n_values)
            ax.plot(g[x_col], rm, color=c, alpha=alpha, label=str(name))

        ax.legend(title=hue)
    else:
        rm = rolling_mean_by_count(d[y_col], n_values)
        ax.plot(d[x_col], rm, label=f"rolling mean (N={n_values})")
        ax.legend()

    ax.set(xlabel=x_col, ylabel=y_col)
    return ax


import os
import glob
import json
import pandas as pd
from stable_baselines3.common.monitor import load_results


def get_standing_clip_test_folders(base_dir="cluster_copy", prefix="standing_clip_test_"):
    """Return all folder names in base_dir that start with prefix."""
    return [
        f for f in os.listdir(base_dir)
        if f.startswith(prefix) and os.path.isdir(os.path.join(base_dir, f))
    ]


def collect_log_and_info_files(base_dir="cluster_copy", prefix="standing_clip_test_"):
    """
    Return lists of:
      - log_files: episode_log.csv OR monitor csv paths (fallback)
      - info_files: run_info.json paths
    (Folder discovery unchanged.)
    """
    folders = get_standing_clip_test_folders(base_dir, prefix)
    log_files, info_files = [], []

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)

        log_path = os.path.join(folder_path, "episode_log.csv")
        info_path = os.path.join(folder_path, "run_info.json")

        if os.path.exists(log_path):
            log_files.append(log_path)
        else:
            # Fallback: find any *.monitor.csv under this folder (keep a CSV path in log_files)
            monitor_csvs = glob.glob(os.path.join(folder_path, "**", "*.monitor.csv"), recursive=True)
            log_files.append(monitor_csvs[0] if monitor_csvs else None)

        if os.path.exists(info_path):
            info_files.append(info_path)

    return log_files, info_files


def _find_monitor_dir_from_run_info(info_file: str) -> str | None:
    """
    If run_info.json contains artifacts.monitor_dir and it exists, return it.
    Accepts Windows-style paths too (\\) because os.path.normpath handles them.
    """
    try:
        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception:
        return None

    monitor_dir = None
    artifacts = info.get("artifacts")
    if isinstance(artifacts, dict):
        monitor_dir = artifacts.get("monitor_dir")

    if not monitor_dir:
        return None

    monitor_dir = os.path.normpath(monitor_dir)
    return monitor_dir if os.path.isdir(monitor_dir) else None


def read_and_combine_logs(log_files, info_files=None):
    """
    Reverted-style signature: takes log_files list.

    New logic:
      - If a valid monitor_dir exists (from run_info.json artifacts.monitor_dir), use SB3 load_results(monitor_dir)
      - Else if it's an episode_log.csv, read with pandas
      - Else if it's a *.monitor.csv but we don't know a monitor_dir, use SB3 load_results(dirname(monitor_csv))
        (preferred over pd.read_csv for monitor files)
    """
    logs = []

    # Build a quick mapping: source_folder -> monitor_dir (if present)
    folder_to_monitor_dir = {}
    if info_files:
        for info_file in info_files:
            source_folder = os.path.basename(os.path.dirname(info_file))
            md = _find_monitor_dir_from_run_info(info_file)
            if md:
                folder_to_monitor_dir[source_folder] = md

    for log_file in log_files:
        if log_file is None:
            continue

        # Identify experiment folder name (same as where run_info.json lives)
        # episode_log.csv path: cluster_copy/<exp>/episode_log.csv
        # monitor csv path: cluster_copy/<exp>/<run_dir>/0.monitor.csv
        p = os.path.normpath(log_file)

        # Heuristic: if episode_log.csv, source is its parent; if monitor csv, source is parent-of-parent
        if p.endswith("episode_log.csv"):
            source_folder = os.path.basename(os.path.dirname(p))
        else:
            source_folder = os.path.basename(os.path.dirname(os.path.dirname(p)))

        # 1) Prefer SB3 monitor_dir if we have it for this experiment
        monitor_dir = folder_to_monitor_dir.get(source_folder)
        if monitor_dir:
            df = load_results(monitor_dir)
            df["source_folder"] = source_folder
            logs.append(df)
            continue

        # 2) Otherwise, use episode_log.csv if that's what we got
        if p.endswith(".csv") and os.path.basename(p) == "episode_log.csv":
            df = pd.read_csv(p)
            df["source_folder"] = source_folder
            logs.append(df)
            continue

        # 3) Otherwise, if it's a monitor csv path, still use SB3 load_results on its directory
        if p.endswith(".monitor.csv"):
            df = load_results(os.path.dirname(p))
            df["source_folder"] = source_folder
            logs.append(df)
            continue

        # 4) Last resort: try reading as CSV
        df = pd.read_csv(p)
        df["source_folder"] = source_folder
        logs.append(df)

    return pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
