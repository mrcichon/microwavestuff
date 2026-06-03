import numpy as np

# Text reports for the regex tab's Stats / Stats-group popups. Curves arrive already
# sorted by regex value and resampled onto a shared grid. Numbers use %g (6 sig figs);
# frequencies use %.6f GHz, matching the reference output. Metric is always |S| in dB.


def _neighbor_diffs(curves):
    # curves: [(name, freq, s_db)]; mean |a-b| in dB between consecutive curves
    lines = []
    diffs = []
    for i in range(len(curves) - 1):
        d = float(np.mean(np.abs(curves[i][2] - curves[i + 1][2])))
        diffs.append(d)
        lines.append(f"{i}-{i+1}: {d:g}   ({curves[i][0]} vs {curves[i+1][0]})")
    return lines, diffs


def format_perfile_range(curves, fmin_ghz, fmax_ghz):
    npts = len(curves[0][1])
    out = ["=== Regex Stats ===",
           f"Range: {fmin_ghz:.6f} .. {fmax_ghz:.6f} GHz   (N={npts})",
           "Metric: |S| [dB]", "",
           "== Mean value per file in selected range =="]
    for name, f, s in curves:
        out.append(f"{name}: {np.mean(s):g}")
    out += ["", "== Minimum per file in selected range =="]
    for name, f, s in curves:
        i = int(np.argmin(s))
        out.append(f"{name}: f_min={f[i]/1e9:.6f} GHz   value_min={s[i]:g}")
    out += ["", "== Mean ABS difference between nearest lines (neighbors) =="]
    lines, diffs = _neighbor_diffs(curves)
    out += lines
    out += ["", f"Global mean |neighbor diff| in range: {np.mean(diffs):g}" if diffs
            else "Global mean |neighbor diff| in range: n/a"]
    return "\n".join(out)


def format_perfile_single(vals, single_ghz):
    out = ["=== Regex Stats ===",
           f"Single frequency: {single_ghz:.6f} GHz",
           "Metric: |S| [dB]", "",
           "== Value per file at selected frequency =="]
    for name, v in vals:
        out.append(f"{name}: {v:g}")
    arr = np.array([v for _, v in vals], dtype=float)
    out += ["", f"Mean across files: {arr.mean():g}", f"Std across files: {arr.std():g}",
            "", "== ABS difference between nearest lines (neighbors) =="]
    for i in range(len(vals) - 1):
        out.append(f"{i}-{i+1}: {abs(vals[i][1]-vals[i+1][1]):g}   ({vals[i][0]} vs {vals[i+1][0]})")
    return "\n".join(out)


def format_group_range(groups, fmin_ghz, fmax_ghz, pattern, group_idx):
    any_member = next(iter(groups.values()))[0]
    npts = len(any_member[1])
    out = ["=== Regex Stats Group ===",
           f"Range: {fmin_ghz:.6f} .. {fmax_ghz:.6f} GHz   (N={npts})",
           "Metric: |S| [dB]",
           f"Pattern: {pattern}   group={group_idx}"]
    for key in sorted(groups):
        members = groups[key]
        out += ["", f"== Group {key} ==", f"Files in group: {len(members)}",
                "Members: " + ", ".join(n for n, _, _ in members)]
        per_means = [float(np.mean(s)) for _, _, s in members]
        out.append(f"Band mean across files: mean={np.mean(per_means):g}   std={np.std(per_means):g}")
        stack = np.array([s for _, _, s in members])
        mean_trace = stack.mean(axis=0)
        j = int(np.argmin(mean_trace))
        f0 = members[0][1]
        out.append(f"Minimum of mean trace: f_min={f0[j]/1e9:.6f} GHz   "
                   f"mean_min={mean_trace[j]:g}   std_at_min={stack[:, j].std():g}")
        out.append("Per-file minima:")
        for name, f, s in members:
            i = int(np.argmin(s))
            out.append(f"  {name}: f_min={f[i]/1e9:.6f} GHz   value_min={s[i]:g}")
    return "\n".join(out)


def format_group_single(groups, single_ghz, pattern, group_idx):
    out = ["=== Regex Stats Group ===",
           f"Single frequency: {single_ghz:.6f} GHz",
           "Metric: |S| [dB]",
           f"Pattern: {pattern}   group={group_idx}"]
    for key in sorted(groups):
        members = groups[key]
        arr = np.array([v for _, v in members], dtype=float)
        out += ["", f"== Group {key} ==", f"Files in group: {len(members)}",
                "Members: " + ", ".join(n for n, _ in members),
                f"Mean at selected frequency: {arr.mean():g}",
                f"Std at selected frequency: {arr.std():g}"]
    return "\n".join(out)
