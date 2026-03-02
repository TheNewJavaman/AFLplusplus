#!/usr/bin/env python3
"""Plot median edge coverage over time for AFL++ baseline vs divergence v5.

Reads 3 trials per group, interpolates to common time grid, computes median
and min/max bands, then plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/gpizarro/dev/mavi/.claude/worktrees/aflpp/evaluation/aflpp-src/div-comparison-work"
N_TRIALS = 3
DURATION = 900  # 15 minutes


def read_plot_data(path):
    """Read AFL++ plot_data file and return (times, edges) arrays."""
    times = []
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            times.append(int(parts[0]))
            edges.append(int(parts[12]))
    return np.array(times), np.array(edges)


def load_trials(group):
    """Load all trials for a group, normalize times to start at 0."""
    trials = []
    for i in range(N_TRIALS):
        path = f"{BASE}/trials/{group}/{i}/output/default/plot_data"
        t, e = read_plot_data(path)
        t = t - t[0]  # normalize to start at 0
        trials.append((t, e))
    return trials


def interpolate_to_grid(trials, grid):
    """Interpolate each trial's edges onto a common time grid."""
    matrix = np.zeros((len(trials), len(grid)))
    for i, (t, e) in enumerate(trials):
        matrix[i] = np.interp(grid, t, e)
    return matrix


# Common time grid: 1-second resolution
grid = np.arange(0, DURATION + 1, 1)

b_trials = load_trials("baseline")
d_trials = load_trials("divergence")

b_matrix = interpolate_to_grid(b_trials, grid)
d_matrix = interpolate_to_grid(d_trials, grid)

b_median = np.median(b_matrix, axis=0)
b_min = np.min(b_matrix, axis=0)
b_max = np.max(b_matrix, axis=0)

d_median = np.median(d_matrix, axis=0)
d_min = np.min(d_matrix, axis=0)
d_max = np.max(d_matrix, axis=0)

# Final stats
print("Baseline   — median edges at end:", int(b_median[-1]),
      f"(range: {int(b_min[-1])}–{int(b_max[-1])})")
print("Divergence — median edges at end:", int(d_median[-1]),
      f"(range: {int(d_min[-1])}–{int(d_max[-1])})")

print("\nPer-trial final edges:")
for group, trials in [("Baseline", b_trials), ("Divergence", d_trials)]:
    finals = sorted([int(e[-1]) for _, e in trials])
    print(f"  {group}: {finals}  (median: {int(np.median(finals))})")

print("\nPer-trial exec/s:")
for group in ["baseline", "divergence"]:
    rates = []
    for i in range(N_TRIALS):
        with open(f"{BASE}/trials/{group}/{i}/output/default/fuzzer_stats") as f:
            for line in f:
                if "execs_per_sec" in line:
                    rates.append(float(line.split(":")[1].strip()))
    print(f"  {group}: {[f'{r:.0f}' for r in rates]}  (median: {np.median(rates):.0f})")

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(grid, b_min, b_max, alpha=0.15, color="#1f77b4")
ax.plot(grid, b_median, label="Baseline (no divergence)", linewidth=2, color="#1f77b4")

ax.fill_between(grid, d_min, d_max, alpha=0.15, color="#d62728")
ax.plot(grid, d_median, label="Divergence v5 (early-out + ref cache)", linewidth=2, color="#d62728")

ax.set_xlabel("Time (seconds)", fontsize=13)
ax.set_ylabel("Edges Found", fontsize=13)
ax.set_title("Edge Coverage Over Time — libpng, minimal seed (15 min, median of 3 trials)", fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_path = f"{BASE}/divergence/coverage_over_time.png"
fig.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
