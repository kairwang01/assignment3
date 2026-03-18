#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSI5388 / ELG5271 - Assignment 3
Phase 1: Basic Security Analysis

What this script does:
1. Optionally tests SSH connectivity to the Hadoop server
2. Optionally retrieves log files via SCP
3. Parses Hadoop / Java-style logs (with fallback support for syslog-like lines)
4. Extracts key details using regex:
   - timestamp
   - log level
   - component / service
   - pid (when available)
   - IP address
   - user (when available)
   - event category
5. Cleans and normalizes the data
6. Performs exploratory analysis and saves graphs
7. Detects anomalies using both rule-based signals and Isolation Forest
8. Exports cleaned data and anomaly results

Example usage:
    python3 phase_1.py --check-ssh --scp
    python3 phase_1.py --local-log-dir ./logs
    python3 phase_1.py --local-log-dir ./logs --no-scp --no-ssh

Default remote settings:
    Server: hadoop.local
    Username: csi5388
    Password: csi5388

Notes:
- This script uses sshpass for non-interactive SSH/SCP.
- On the VM, install it if needed:
      sudo apt-get update && sudo apt-get install -y sshpass
- If your actual logs have a slightly different format, adjust the regex patterns below.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Configuration
# ----------------------------

DEFAULT_HOST = "hadoop.local"
DEFAULT_USER = "csi5388"
DEFAULT_PASSWORD = "csi5388"
DEFAULT_REMOTE_LOG_DIR = "/opt/hadoop/logs"
DEFAULT_LOCAL_LOG_DIR = "./phase1_logs"
DEFAULT_OUTPUT_DIR = "./phase1_output"

# Hadoop / Java log example:
# 2025-03-18 10:23:45,123 INFO org.apache.hadoop.hdfs.server.namenode.NameNode: Message...
JAVA_LOG_RE = re.compile(
    r"""
    ^
    (?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})
    \s+
    (?P<level>TRACE|DEBUG|INFO|WARN|ERROR|FATAL)
    \s+
    (?P<component>[\w.$/-]+)
    \s*:\s*
    (?P<message>.*)
    $
    """,
    re.VERBOSE,
)

# Syslog-style fallback:
# Mar 18 10:23:45 host sshd[1234]: Failed password for root from 192.168.1.1 port 22 ssh2
SYSLOG_RE = re.compile(
    r"""
    ^
    (?P<timestamp>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})
    \s+
    (?P<host>\S+)
    \s+
    (?P<service>[\w./-]+)(?:\[(?P<pid>\d+)\])?
    :
    \s*
    (?P<message>.*)
    $
    """,
    re.VERBOSE,
)

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
USER_RE = re.compile(
    r"""
    \b
    (?:
        user\s*[=:]?\s* |
        username\s*[=:]?\s* |
        for\s+ |
        from\s+
    )
    (?P<user>[A-Za-z0-9._-]+)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Common event patterns for basic security analysis
EVENT_PATTERNS = {
    "authentication_failure": re.compile(
        r"failed|failure|invalid password|authentication failed|denied",
        re.IGNORECASE,
    ),
    "authentication_success": re.compile(
        r"accepted|success|authenticated|logged in|login successful",
        re.IGNORECASE,
    ),
    "network_connection": re.compile(
        r"connection|connected|disconnected|socket|handshake|bind|listen",
        re.IGNORECASE,
    ),
    "file_system_event": re.compile(
        r"hdfs|namenode|datanode|block|replica|filesystem|fsimage|edits",
        re.IGNORECASE,
    ),
    "error_exception": re.compile(
        r"exception|error|fatal|stack trace|traceback",
        re.IGNORECASE,
    ),
    "warning_event": re.compile(
        r"warn|warning|retry|timeout",
        re.IGNORECASE,
    ),
}


# ----------------------------
# Helpers
# ----------------------------

@dataclass
class RemoteConfig:
    host: str
    username: str
    password: str
    remote_log_dir: str
    local_log_dir: str


def run_shell_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the completed process."""
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def test_ssh_connection(cfg: RemoteConfig) -> bool:
    """
    Test SSH connectivity to satisfy the marking requirement.
    Requires sshpass installed.
    """
    print(f"[INFO] Testing SSH connection to {cfg.username}@{cfg.host} ...")
    cmd = [
        "sshpass", "-p", cfg.password,
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{cfg.username}@{cfg.host}",
        "echo SSH_CONNECTION_OK"
    ]
    try:
        result = run_shell_command(cmd, check=True)
        ok = "SSH_CONNECTION_OK" in result.stdout
        if ok:
            print("[INFO] SSH connection successful.")
        else:
            print("[WARN] SSH command ran, but expected confirmation text not found.")
        return ok
    except Exception as exc:
        print(f"[ERROR] SSH connection failed: {exc}")
        return False


def retrieve_logs_via_scp(cfg: RemoteConfig) -> int:
    """
    Retrieve logs from remote server to local directory using SCP.
    Requires sshpass installed.
    """
    ensure_dir(cfg.local_log_dir)
    remote_glob = f"{cfg.username}@{cfg.host}:{cfg.remote_log_dir}/*.log"
    print(f"[INFO] Copying logs from {remote_glob} to {cfg.local_log_dir} ...")

    cmd = [
        "sshpass", "-p", cfg.password,
        "scp",
        "-o", "StrictHostKeyChecking=no",
        remote_glob,
        cfg.local_log_dir
    ]

    try:
        result = run_shell_command(cmd, check=True)
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())

        files = glob.glob(os.path.join(cfg.local_log_dir, "*.log"))
        print(f"[INFO] Retrieved {len(files)} .log file(s).")
        return len(files)
    except Exception as exc:
        print(f"[ERROR] SCP retrieval failed: {exc}")
        return 0


def classify_event(message: str) -> str:
    """Assign a broad event category based on message content."""
    if not isinstance(message, str):
        return "unknown"

    for label, pattern in EVENT_PATTERNS.items():
        if pattern.search(message):
            return label
    return "other"


def extract_ip(text: str) -> Optional[str]:
    """Extract first IPv4 address if present."""
    if not isinstance(text, str):
        return None
    match = IP_RE.search(text)
    return match.group(0) if match else None


def extract_user(text: str) -> Optional[str]:
    """Extract probable username if present."""
    if not isinstance(text, str):
        return None
    match = USER_RE.search(text)
    if match:
        return match.group("user")
    return None


def parse_log_line(line: str, source_file: str) -> dict:
    """
    Parse one line into a normalized record.
    Supports Java/Hadoop logs first, then syslog-style logs, then fallback.
    """
    line = line.rstrip("\n")

    record = {
        "source_file": source_file,
        "raw_line": line,
        "format_type": "unparsed",
        "timestamp": None,
        "log_level": None,
        "component": None,
        "host": None,
        "service": None,
        "pid": None,
        "message": line,
        "ip": extract_ip(line),
        "user": extract_user(line),
        "event_type": classify_event(line),
    }

    m_java = JAVA_LOG_RE.match(line)
    if m_java:
        gd = m_java.groupdict()
        message = gd["message"]
        record.update({
            "format_type": "java_hadoop",
            "timestamp": gd["timestamp"],
            "log_level": gd["level"],
            "component": gd["component"],
            "message": message,
            "ip": extract_ip(message) or record["ip"],
            "user": extract_user(message) or record["user"],
            "event_type": classify_event(message),
        })
        return record

    m_sys = SYSLOG_RE.match(line)
    if m_sys:
        gd = m_sys.groupdict()
        message = gd["message"]
        record.update({
            "format_type": "syslog",
            "timestamp": gd["timestamp"],
            "host": gd["host"],
            "service": gd["service"],
            "pid": gd["pid"],
            "component": gd["service"],
            "message": message,
            "ip": extract_ip(message) or record["ip"],
            "user": extract_user(message) or record["user"],
            "event_type": classify_event(message),
        })

        # infer log level from syslog message when possible
        upper_msg = message.upper()
        if "ERROR" in upper_msg:
            record["log_level"] = "ERROR"
        elif "WARN" in upper_msg or "WARNING" in upper_msg:
            record["log_level"] = "WARN"
        elif "DEBUG" in upper_msg:
            record["log_level"] = "DEBUG"
        elif "INFO" in upper_msg:
            record["log_level"] = "INFO"
        return record

    return record


def load_and_parse_logs(local_log_dir: str) -> pd.DataFrame:
    """Load all .log files and parse them into a DataFrame."""
    files = sorted(glob.glob(os.path.join(local_log_dir, "*.log")))
    if not files:
        raise FileNotFoundError(f"No .log files found in: {local_log_dir}")

    rows: list[dict] = []
    for file_path in files:
        source_file = os.path.basename(file_path)
        print(f"[INFO] Parsing {source_file} ...")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    rows.append(parse_log_line(line, source_file))

    df = pd.DataFrame(rows)
    return df


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize timestamp strings into pandas datetime.
    Supports:
    - Java/Hadoop: YYYY-MM-DD HH:MM:SS,mmm
    - Syslog-like: Mon DD HH:MM:SS (year will default to current year)
    """
    df = df.copy()

    ts_java = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S,%f", errors="coerce")
    ts_sys = pd.to_datetime(df["timestamp"], format="%b %d %H:%M:%S", errors="coerce")

    # If syslog timestamps are used, pandas assigns 1900 as year; replace with current year
    current_year = pd.Timestamp.now().year
    ts_sys = ts_sys.apply(
        lambda x: x.replace(year=current_year) if pd.notna(x) else pd.NaT
    )

    df["timestamp_dt"] = ts_java.fillna(ts_sys)
    return df


def clean_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize logs:
    - remove exact duplicates
    - standardize missing strings
    - normalize log levels
    - add helper fields
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=["source_file", "raw_line"]).reset_index(drop=True)
    after = len(df)
    print(f"[INFO] Removed {before - after} duplicate rows.")

    # Standardize string columns
    string_cols = [
        "format_type", "log_level", "component", "host", "service",
        "message", "ip", "user", "event_type", "source_file"
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    df["log_level"] = df["log_level"].fillna("UNKNOWN").str.upper()
    df["event_type"] = df["event_type"].fillna("unknown")

    # Normalize fields used later
    df["has_ip"] = df["ip"].notna().astype(int)
    df["has_user"] = df["user"].notna().astype(int)
    df["is_error_level"] = df["log_level"].isin(["ERROR", "FATAL"]).astype(int)
    df["is_warn_level"] = df["log_level"].isin(["WARN", "WARNING"]).astype(int)

    # Time helpers
    df = normalize_timestamps(df)
    df["hour"] = df["timestamp_dt"].dt.hour
    df["date"] = df["timestamp_dt"].dt.date.astype("string")
    df["minute_bucket"] = df["timestamp_dt"].dt.floor("min")

    return df


def save_basic_outputs(df: pd.DataFrame, output_dir: str) -> None:
    """Save cleaned logs and summary tables."""
    ensure_dir(output_dir)

    cleaned_path = os.path.join(output_dir, "phase1_cleaned_logs.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"[INFO] Saved cleaned logs to: {cleaned_path}")

    # Basic summaries
    summary = {
        "total_rows": [len(df)],
        "parsed_rows": [(df["format_type"] != "unparsed").sum()],
        "unparsed_rows": [(df["format_type"] == "unparsed").sum()],
        "unique_source_files": [df["source_file"].nunique()],
        "unique_components": [df["component"].nunique(dropna=True)],
        "unique_ips": [df["ip"].nunique(dropna=True)],
        "unique_users": [df["user"].nunique(dropna=True)],
    }
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, "phase1_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Saved summary to: {summary_path}")

    # Frequency tables
    for col in ["log_level", "event_type", "component", "source_file"]:
        freq = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
        freq.to_csv(os.path.join(output_dir, f"{col}_counts.csv"), index=False)


def plot_and_save(df: pd.DataFrame, output_dir: str) -> None:
    """Generate exploratory visualizations required by the rubric."""
    ensure_dir(output_dir)

    # 1. Log level distribution
    plt.figure(figsize=(8, 5))
    df["log_level"].value_counts().plot(kind="bar")
    plt.title("Log Level Distribution")
    plt.xlabel("Log Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_log_level_distribution.png"), dpi=200)
    plt.close()

    # 2. Event type distribution
    plt.figure(figsize=(10, 5))
    df["event_type"].value_counts().plot(kind="bar")
    plt.title("Event Type Distribution")
    plt.xlabel("Event Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_event_type_distribution.png"), dpi=200)
    plt.close()

    # 3. Top components
    top_components = df["component"].value_counts().head(15)
    if not top_components.empty:
        plt.figure(figsize=(10, 6))
        top_components.sort_values().plot(kind="barh")
        plt.title("Top 15 Components / Services")
        plt.xlabel("Count")
        plt.ylabel("Component")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_top_components.png"), dpi=200)
        plt.close()

    # 4. Timeline by minute
    if df["minute_bucket"].notna().any():
        ts_counts = df.groupby("minute_bucket").size()
        plt.figure(figsize=(12, 5))
        ts_counts.plot()
        plt.title("Log Volume Over Time")
        plt.xlabel("Time")
        plt.ylabel("Number of Log Entries")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_log_volume_over_time.png"), dpi=200)
        plt.close()

    # 5. Hourly distribution
    if df["hour"].notna().any():
        plt.figure(figsize=(8, 5))
        df["hour"].dropna().astype(int).value_counts().sort_index().plot(kind="bar")
        plt.title("Hourly Log Activity")
        plt.xlabel("Hour of Day")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_hourly_activity.png"), dpi=200)
        plt.close()


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregated features for anomaly detection.
    We detect anomalies at the minute level because single raw log lines
    are often too noisy and less informative than short time-window behavior.
    """
    if df["minute_bucket"].notna().sum() == 0:
        # fallback: assign synthetic bucket from row index
        tmp = df.copy()
        tmp["minute_bucket"] = pd.date_range(
            start=pd.Timestamp.now().floor("min"),
            periods=len(tmp),
            freq="min"
        )

    grouped = (
        df.groupby("minute_bucket")
        .agg(
            total_logs=("raw_line", "count"),
            unique_components=("component", lambda s: s.dropna().nunique()),
            unique_ips=("ip", lambda s: s.dropna().nunique()),
            unique_users=("user", lambda s: s.dropna().nunique()),
            error_logs=("is_error_level", "sum"),
            warn_logs=("is_warn_level", "sum"),
            with_ip=("has_ip", "sum"),
            with_user=("has_user", "sum"),
        )
        .reset_index()
    )

    # count event types separately when present
    for evt in [
        "authentication_failure",
        "authentication_success",
        "network_connection",
        "file_system_event",
        "error_exception",
        "warning_event",
        "other",
    ]:
        evt_counts = (
            df.assign(evt_match=(df["event_type"] == evt).astype(int))
            .groupby("minute_bucket")["evt_match"]
            .sum()
            .reset_index(name=f"evt_{evt}")
        )
        grouped = grouped.merge(evt_counts, on="minute_bucket", how="left")

    grouped = grouped.fillna(0)

    # ratios
    grouped["error_ratio"] = grouped["error_logs"] / grouped["total_logs"].replace(0, 1)
    grouped["warn_ratio"] = grouped["warn_logs"] / grouped["total_logs"].replace(0, 1)
    grouped["ip_ratio"] = grouped["with_ip"] / grouped["total_logs"].replace(0, 1)
    grouped["user_ratio"] = grouped["with_user"] / grouped["total_logs"].replace(0, 1)

    return grouped


def detect_anomalies(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolation Forest anomaly detection.
    Higher anomaly_score means more suspicious after inversion.
    """
    feature_df = feature_df.copy()

    numeric_cols = [c for c in feature_df.columns if c != "minute_bucket"]
    X = feature_df[numeric_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # sklearn score_samples: lower means more abnormal
    raw_scores = model.score_samples(X_scaled)
    preds = model.predict(X_scaled)  # -1 anomaly, 1 normal

    feature_df["iso_raw_score"] = raw_scores
    feature_df["is_anomaly"] = (preds == -1).astype(int)

    # invert for easier interpretation: larger = more suspicious
    feature_df["anomaly_score"] = -feature_df["iso_raw_score"]

    # add simple rule-based flags for interpretability
    feature_df["rule_high_errors"] = (
        feature_df["error_logs"] >= feature_df["error_logs"].quantile(0.95)
    ).astype(int)

    feature_df["rule_high_volume"] = (
        feature_df["total_logs"] >= feature_df["total_logs"].quantile(0.95)
    ).astype(int)

    feature_df["rule_many_unique_ips"] = (
        feature_df["unique_ips"] >= feature_df["unique_ips"].quantile(0.95)
    ).astype(int)

    feature_df["rule_many_unique_components"] = (
        feature_df["unique_components"] >= feature_df["unique_components"].quantile(0.95)
    ).astype(int)

    feature_df["rule_flag_count"] = feature_df[
        ["rule_high_errors", "rule_high_volume", "rule_many_unique_ips", "rule_many_unique_components"]
    ].sum(axis=1)

    # prioritize events that are both ML anomalies and match multiple rules
    feature_df["priority_score"] = feature_df["anomaly_score"] + 0.25 * feature_df["rule_flag_count"]

    return feature_df.sort_values(["is_anomaly", "priority_score"], ascending=[False, False]).reset_index(drop=True)


def attach_raw_context(df: pd.DataFrame, anomaly_df: pd.DataFrame, output_dir: str) -> None:
    """
    Save contextual raw logs around anomalous minute buckets to support investigation.
    """
    anomaly_minutes = anomaly_df.loc[anomaly_df["is_anomaly"] == 1, "minute_bucket"].dropna().tolist()
    if not anomaly_minutes:
        print("[INFO] No anomaly windows detected by Isolation Forest.")
        return

    context_rows = []
    for ts in anomaly_minutes:
        subset = df[df["minute_bucket"] == ts].copy()
        subset["anomaly_minute"] = ts
        context_rows.append(subset)

    context_df = pd.concat(context_rows, ignore_index=True) if context_rows else pd.DataFrame()
    context_path = os.path.join(output_dir, "phase1_anomaly_context_logs.csv")
    context_df.to_csv(context_path, index=False)
    print(f"[INFO] Saved anomaly context logs to: {context_path}")


def print_console_summary(df: pd.DataFrame, anomaly_df: pd.DataFrame) -> None:
    """Print concise findings for quick inspection."""
    print("\n========== PHASE 1 SUMMARY ==========")
    print(f"Total parsed log rows: {len(df)}")
    print(f"Parsed rows (recognized format): {(df['format_type'] != 'unparsed').sum()}")
    print(f"Unique files: {df['source_file'].nunique()}")
    print(f"Unique components: {df['component'].nunique(dropna=True)}")
    print(f"Unique IPs: {df['ip'].nunique(dropna=True)}")
    print(f"Unique users: {df['user'].nunique(dropna=True)}")

    print("\nTop log levels:")
    print(df["log_level"].value_counts().head(10).to_string())

    print("\nTop event types:")
    print(df["event_type"].value_counts().head(10).to_string())

    if anomaly_df["is_anomaly"].sum() > 0:
        print("\nTop anomaly windows:")
        cols = [
            "minute_bucket", "total_logs", "error_logs", "unique_components",
            "unique_ips", "unique_users", "anomaly_score", "rule_flag_count"
        ]
        print(anomaly_df.loc[anomaly_df["is_anomaly"] == 1, cols].head(10).to_string(index=False))
    else:
        print("\nNo anomaly windows flagged by the current model settings.")


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 - Basic Security Analysis")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--username", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--remote-log-dir", default=DEFAULT_REMOTE_LOG_DIR)
    parser.add_argument("--local-log-dir", default=DEFAULT_LOCAL_LOG_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--check-ssh", action="store_true", help="Test SSH connectivity first")
    parser.add_argument("--scp", action="store_true", help="Retrieve logs from remote host using SCP")
    parser.add_argument("--no-ssh", action="store_true", help="Skip SSH check")
    parser.add_argument("--no-scp", action="store_true", help="Skip SCP retrieval")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = RemoteConfig(
        host=args.host,
        username=args.username,
        password=args.password,
        remote_log_dir=args.remote_log_dir,
        local_log_dir=args.local_log_dir,
    )

    ensure_dir(args.local_log_dir)
    ensure_dir(args.output_dir)

    # SSH connection check
    if args.check_ssh and not args.no_ssh:
        test_ssh_connection(cfg)

    # SCP retrieval
    if args.scp and not args.no_scp:
        retrieve_logs_via_scp(cfg)

    # Load, parse, clean
    df = load_and_parse_logs(args.local_log_dir)
    df = clean_logs(df)
    save_basic_outputs(df, args.output_dir)
    plot_and_save(df, args.output_dir)

    # Anomaly detection
    feature_df = build_anomaly_features(df)
    anomaly_df = detect_anomalies(feature_df)

    anomaly_path = os.path.join(args.output_dir, "phase1_anomaly_windows.csv")
    anomaly_df.to_csv(anomaly_path, index=False)
    print(f"[INFO] Saved anomaly windows to: {anomaly_path}")

    attach_raw_context(df, anomaly_df, args.output_dir)
    print_console_summary(df, anomaly_df)

    print("\n[INFO] Phase 1 analysis completed successfully.")


if __name__ == "__main__":
    main()
