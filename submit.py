import argparse
import csv
import datetime
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

BASE_URL = "http://34.63.153.158"
TASK_ID = "22-forging-task"


def submit(zip_path, budget, wait=False, poll_interval=20, max_attempts=200):
    api_key = os.environ["WM_API_KEY"]
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert sorted(names, key=lambda s: int(s[:-4])) == [f"{i}.png" for i in range(1, 201)]
        assert all("/" not in n for n in names)
    print("ZIP OK:", zip_path, flush=True)

    for attempt in range(max_attempts if wait else 1):
        with open(zip_path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": api_key},
                files={"file": (zip_path.name, f, "application/zip")},
                timeout=120,
            )
        if r.status_code == 429 and wait:
            detail = r.json().get("detail", "") if r.headers.get("content-type", "").startswith("application/json") else ""
            print(f"attempt {attempt}: 429 cooldown ({detail}), retrying in {poll_interval}s", flush=True)
            time.sleep(poll_interval)
            continue
        break

    print("HTTP", r.status_code, flush=True)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    print("RESULT:", body, flush=True)

    with open("results_log.csv", "a", newline="") as fp:
        csv.writer(fp).writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            zip_path.stem, math.exp(-8 * budget), r.status_code, body,
        ])
    return r.status_code, body


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("--budget", type=float, default=0.022, help="for the results_log Sqlt record only")
    parser.add_argument("--wait", action="store_true", help="retry every 20s until the cooldown clears")
    args = parser.parse_args()
    status, body = submit(args.zip_path, args.budget, wait=args.wait)
    sys.exit(0 if status == 200 else 1)
