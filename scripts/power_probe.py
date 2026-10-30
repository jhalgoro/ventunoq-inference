#!/usr/bin/env python3
"""INA232 power + SoC temperature probe (sudo-free) — L4 / ROADMAP L4.

Samples /sys/class/hwmon/hwmon2 (name: ina232) power1_input (µW) plus every temp*_input
under /sys/class/hwmon (hwmon2 itself has NO temp files — temps live in the separate
nsp_*/cpuss_*/ddrss_*/cpu_* thermal hwmons; verified 2026-10-29). Run it alongside any
bench suite; stop with Ctrl-C (or send SIGTERM). Output: results/power_<tag>.csv, one
row per sample, plus a summary line (avg/max W, peak °C per zone group) on exit.

J/token (the headline metric) is computed post-hoc: J/tok = avg_W ÷ t/s from the bench
log — keep the two timestamps aligned by starting the probe first.

Usage:  scripts/power_probe.py <tag> [hz]        # e.g. scripts/power_probe.py lfm3_npu 5
"""
import csv
import glob
import os
import signal
import sys
import time
from datetime import datetime, timezone

POWER_GLOB = "/sys/class/hwmon/hwmon*/power1_input"   # exactly one match expected (ina232)


def find_ina232():
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(d + "/name").read().strip() == "ina232":
                return d
        except OSError:
            pass
    sys.exit("no ina232 hwmon found — board/kernel changed? (verified 2026-10-29: hwmon2)")


def temp_sensors():
    """(label, path) for every temp*_input; label = <hwmon name>.<tempN>."""
    out = []
    for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            name = open(d + "/name").read().strip()
        except OSError:
            continue
        for t in sorted(glob.glob(d + "/temp*_input")):
            label = name + "." + os.path.basename(t).split("_")[0]  # e.g. cpuss_1_0_thermal.temp1
            out.append((label, t))
    return out


def read(path):
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return None


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    hz = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    ina = find_ina232()
    sensors = temp_sensors()
    interesting = [s for s in sensors
                   if any(k in s[0] for k in ("nsp", "cpuss", "ddrss", "cpu_"))]
    # keep CSV width sane: all temps would be 46 columns; log interesting zones + global max
    out = f"results/power_{tag}.csv"
    os.makedirs("results", exist_ok=True)
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["ts", "watts"] + [lbl for lbl, _ in interesting] + ["temp_max_c"])

    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))

    e_max = 0.0          # energy integrator (J)
    t_prev = None
    n = 0
    peak = {}
    tmax_all = 0.0
    print(f"probe: {ina} + {len(interesting)} zones → {out} @ {hz} Hz (Ctrl-C to stop)")
    while not stop:
        t0 = time.monotonic()
        p_uw = read(ina + "/power1_input")
        if p_uw is None:
            stop.append(1)
            break
        watts = p_uw / 1e6
        # temp*_input are millidegrees C; power1_input is µW (verified 2026-10-29)
        temps = {lbl: (v / 1000.0 if (v := read(p)) is not None else None)
                 for lbl, p in interesting}
        tmax = max([v for v in temps.values() if v] or [0])
        if t_prev is not None:
            e_max += watts * (t0 - t_prev)
        t_prev = t0
        n += 1
        for lbl, v in temps.items():
            if v and v > peak.get(lbl, 0):
                peak[lbl] = v
        tmax_all = max(tmax_all, tmax)
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    f"{watts:.3f}"] + [temps[lbl] if temps[lbl] is not None else ""
                                       for lbl, _ in interesting] + [tmax])
        if n % int(hz * 10) == 0:      # heartbeat every 10 s
            print(f"  {watts:6.2f} W  tmax {tmax} °C  ({n} samples)")
        fh.flush()
        time.sleep(max(0.0, 1.0 / hz - (time.monotonic() - t0)))

    fh.close()
    print(f"\n{out}: {n} samples, {e_max:.0f} J drawn, peak {tmax_all:.1f} °C")
    print("peaks per zone: " + ", ".join(f"{k}={v}" for k, v in sorted(peak.items())))
    print("J/token = avg_W / t/s — pair with the bench log's mean t/s (L4).")


if __name__ == "__main__":
    main()
