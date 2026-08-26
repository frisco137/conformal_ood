#!/usr/bin/env python3
"""Run all mechanistic interpretability hook sanity checks.

Sequentially runs the hook demo for each of the four models and prints
a combined summary table with pass/fail status.

Usage:
    python models/mech_interp_suite/run_all_sanity_checks.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

_models_dir = str(Path(__file__).resolve().parent.parent)
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
_project_dir = str(Path(__file__).resolve().parent.parent.parent)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def main():
    results = []

    # --- TabPFN v3 ---
    try:
        print("\n" + "#" * 80)
        print("# [1/4] TabPFN v3")
        print("#" * 80)
        t0 = time.time()
        from mech_interp_suite.hooks_tabpfn_v3 import run_hooks_tabpfn_v3
        acts, ok = run_hooks_tabpfn_v3()
        dt = time.time() - t0
        results.append(("TabPFN v3", len(acts), ok, f"{dt:.1f}s", ""))
    except Exception as e:
        results.append(("TabPFN v3", 0, False, "N/A", str(e)))
        traceback.print_exc()

    # --- TabPFN v2 ---
    try:
        print("\n" + "#" * 80)
        print("# [2/4] TabPFN v2")
        print("#" * 80)
        t0 = time.time()
        from mech_interp_suite.hooks_tabpfn_v2 import run_hooks_tabpfn_v2
        acts, ok = run_hooks_tabpfn_v2()
        dt = time.time() - t0
        results.append(("TabPFN v2", len(acts), ok, f"{dt:.1f}s", ""))
    except Exception as e:
        results.append(("TabPFN v2", 0, False, "N/A", str(e)))
        traceback.print_exc()

    # --- TabICL v2 ---
    try:
        print("\n" + "#" * 80)
        print("# [3/4] TabICL v2")
        print("#" * 80)
        t0 = time.time()
        from mech_interp_suite.hooks_tabicl_v2 import run_hooks_tabicl_v2
        acts, ok = run_hooks_tabicl_v2()
        dt = time.time() - t0
        results.append(("TabICL v2", len(acts), ok, f"{dt:.1f}s", ""))
    except Exception as e:
        results.append(("TabICL v2", 0, False, "N/A", str(e)))
        traceback.print_exc()

    # --- Google TabFM ---
    try:
        print("\n" + "#" * 80)
        print("# [4/4] Google TabFM")
        print("#" * 80)
        t0 = time.time()
        from mech_interp_suite.hooks_google_tabfm import run_hooks_google_tabfm
        acts, ok = run_hooks_google_tabfm()
        dt = time.time() - t0
        results.append(("Google TabFM", len(acts), ok, f"{dt:.1f}s", ""))
    except Exception as e:
        results.append(("Google TabFM", 0, False, "N/A", str(e)))
        traceback.print_exc()

    # --- Summary ---
    sep = "=" * 90
    print(f"\n\n{sep}")
    print("  MECHANISTIC INTERPRETABILITY SUITE — SANITY CHECK SUMMARY")
    print(sep)
    print(f"  {'Model':<16} {'# Hooks':<10} {'Time':<10} {'Status':<10} {'Error'}")
    print("-" * 90)
    all_ok = True
    for name, n_hooks, ok, dt, err in results:
        status = "PASSED" if ok else "FAILED"
        if not ok:
            all_ok = False
        err_short = err[:40] + "..." if len(err) > 40 else err
        print(f"  {name:<16} {n_hooks:<10} {dt:<10} {status:<10} {err_short}")
    print(sep)

    if all_ok:
        print("  ALL SANITY CHECKS PASSED!")
    else:
        print("  SOME CHECKS FAILED — see details above.")
    print(sep)

    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
