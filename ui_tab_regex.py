import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import re
import sys

from analysis_regex import (extract_regex_value, find_best_drop, compute_kendall_tau,
                            compute_max_displacement, mask_to_ranges, compute_small_diffs,
                            compute_congruence, congruence_template, mask_to_index_ranges,
                            compute_shift_tracking, format_regex_text)
from analysis_regex_stats import (format_perfile_range, format_perfile_single,
                                  format_group_range, format_group_single)


class TabRegex:
    def __init__(self, parent, control_frame, fig, canvas,
                 legend_frame, legend_canvas,
                 get_files_func, get_freq_range_func,
                 get_legend_on_plot_func, get_scale_mode_func):
        self.parent = parent
        self.control_frame = control_frame
        self.fig = fig
        self.canvas = canvas
        self.legend_frame = legend_frame
        self.legend_canvas = legend_canvas
        self.get_files = get_files_func
        self.get_freq_range = get_freq_range_func
        self.get_legend_on_plot = get_legend_on_plot_func
        self.get_scale_mode = get_scale_mode_func

        self.regex_pattern = tk.StringVar(value=r'_(\d+)ml')
        self.regex_group = tk.IntVar(value=1)
        self.regex_param = tk.StringVar(value="s21")
        self.regex_highlight = tk.BooleanVar(value=True)
        self.regex_strict = tk.BooleanVar(value=False)
        self.regex_n_drop = tk.IntVar(value=0)
        self.regex_small_diff = tk.BooleanVar(value=False)
        self.regex_small_threshold = tk.DoubleVar(value=0.1)
        self.regex_small_ref = tk.DoubleVar(value=10.0)
        self.regex_small_count = tk.IntVar(value=1)
        self.regex_gate = tk.BooleanVar(value=False)
        self.regex_gate_center = tk.DoubleVar(value=5.0)
        self.regex_gate_span = tk.DoubleVar(value=0.5)
        self.regex_phase = tk.BooleanVar(value=False)
        self.regex_tau = tk.BooleanVar(value=False)
        self.regex_tau_threshold = tk.DoubleVar(value=0.8)
        self.regex_disp = tk.BooleanVar(value=False)
        self.regex_disp_threshold = tk.IntVar(value=1)
        self.regex_congru = tk.BooleanVar(value=False)
        self.regex_congru_mode = tk.StringVar(value="vertical")
        self.regex_congru_window = tk.IntVar(value=21)
        self.regex_congru_threshold = tk.DoubleVar(value=0.9)
        self.regex_congru_maxlag = tk.IntVar(value=10)
        self.regex_track = tk.BooleanVar(value=False)
        self.regex_track_mono = tk.DoubleVar(value=0.9)

        self.regex_ranges = []
        self.last_result = None

        self._build_ui()

    def _build_ui(self):
        frame = ttk.Frame(self.control_frame)
        frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Label(frame, text="Pattern:").pack(side=tk.LEFT, padx=(0, 5))
        self.regex_entry = ttk.Entry(frame, textvariable=self.regex_pattern, width=20)
        self.regex_entry.pack(side=tk.LEFT, padx=2)
        self.regex_entry.bind("<KeyRelease>", self._on_regex_change)
        self.regex_entry.bind("<Return>", lambda e: self.update())

        ttk.Label(frame, text="Group:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(frame, from_=1, to=10, width=3,
                     textvariable=self.regex_group, command=self.update).pack(side=tk.LEFT, padx=2)

        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(frame, text="Param:").pack(side=tk.LEFT, padx=(0, 5))
        for n in ("s11", "s12", "s21", "s22"):
            ttk.Radiobutton(frame, text=n.upper(), value=n,
                            variable=self.regex_param, command=self.update).pack(side=tk.LEFT, padx=2)

        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Checkbutton(frame, text="Gating", variable=self.regex_gate,
                         command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame, text="Center[ns]").pack(side=tk.LEFT, padx=(5, 2))
        gate_center_entry = ttk.Entry(frame, textvariable=self.regex_gate_center, width=5)
        gate_center_entry.pack(side=tk.LEFT, padx=2)
        gate_center_entry.bind("<Return>", lambda e: self.update())
        ttk.Label(frame, text="Span[ns]").pack(side=tk.LEFT, padx=(5, 2))
        gate_span_entry = ttk.Entry(frame, textvariable=self.regex_gate_span, width=5)
        gate_span_entry.pack(side=tk.LEFT, padx=2)
        gate_span_entry.bind("<Return>", lambda e: self.update())

        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Checkbutton(frame, text="Highlight monotonic",
                         variable=self.regex_highlight, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(frame, text="Strict",
                         variable=self.regex_strict, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame, text="Drop:").pack(side=tk.LEFT, padx=(5, 2))
        ttk.Spinbox(frame, from_=0, to=20, width=3,
                     textvariable=self.regex_n_drop, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(frame, text="Phase",
                         variable=self.regex_phase, command=self.update).pack(side=tk.LEFT, padx=2)

        ttk.Button(frame, text="Export", command=self._export_ranges).pack(side=tk.LEFT, padx=10)
        ttk.Button(frame, text="Stats", command=self._open_stats).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Stats group", command=self._open_stats_group).pack(side=tk.LEFT, padx=2)

        frame2 = ttk.Frame(self.control_frame)
        frame2.pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Checkbutton(frame2, text="Small diffs",
                         variable=self.regex_small_diff, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame2, text="Tol:").pack(side=tk.LEFT, padx=(5, 2))
        small_tol_entry = ttk.Entry(frame2, textvariable=self.regex_small_threshold, width=5)
        small_tol_entry.pack(side=tk.LEFT, padx=2)
        small_tol_entry.bind("<Return>", lambda e: self.update())
        ttk.Label(frame2, text="@").pack(side=tk.LEFT, padx=2)
        small_ref_entry = ttk.Entry(frame2, textvariable=self.regex_small_ref, width=5)
        small_ref_entry.pack(side=tk.LEFT, padx=2)
        small_ref_entry.bind("<Return>", lambda e: self.update())
        ttk.Label(frame2, text="dB").pack(side=tk.LEFT, padx=(0, 10))

        ttk.Separator(frame2, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Checkbutton(frame2, text="Kendall tau",
                         variable=self.regex_tau, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame2, text="|tau|>=").pack(side=tk.LEFT, padx=(2, 0))
        tau_entry = ttk.Entry(frame2, textvariable=self.regex_tau_threshold, width=4)
        tau_entry.pack(side=tk.LEFT, padx=2)
        tau_entry.bind("<Return>", lambda e: self.update())

        ttk.Separator(frame2, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Checkbutton(frame2, text="Max disp.",
                         variable=self.regex_disp, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame2, text="<=").pack(side=tk.LEFT, padx=(2, 0))
        disp_entry = ttk.Entry(frame2, textvariable=self.regex_disp_threshold, width=3)
        disp_entry.pack(side=tk.LEFT, padx=2)
        disp_entry.bind("<Return>", lambda e: self.update())

        ttk.Separator(frame2, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(frame2, text="Quick:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(frame2, text="_XXml", width=6,
                   command=lambda: self._set_pattern(r'_(\d+)ml')).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame2, text="XXmm", width=6,
                   command=lambda: self._set_pattern(r'(\d+)mm')).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame2, text="XXcm", width=6,
                   command=lambda: self._set_pattern(r'(\d+)cm')).pack(side=tk.LEFT, padx=2)

        frame3 = ttk.Frame(self.control_frame)
        frame3.pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Checkbutton(frame3, text="Shared shape",
                         variable=self.regex_congru, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(frame3, text="Vert", value="vertical",
                        variable=self.regex_congru_mode, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(frame3, text="Freq-shift", value="horizontal",
                        variable=self.regex_congru_mode, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame3, text="Window:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(frame3, from_=3, to=201, increment=2, width=4,
                     textvariable=self.regex_congru_window, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame3, text="Congr>=").pack(side=tk.LEFT, padx=(8, 0))
        congru_entry = ttk.Entry(frame3, textvariable=self.regex_congru_threshold, width=5)
        congru_entry.pack(side=tk.LEFT, padx=2)
        congru_entry.bind("<Return>", lambda e: self.update())
        ttk.Label(frame3, text="Max shift:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(frame3, from_=1, to=200, width=4,
                     textvariable=self.regex_congru_maxlag, command=self.update).pack(side=tk.LEFT, padx=2)

        ttk.Separator(frame3, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Checkbutton(frame3, text="Track shift",
                         variable=self.regex_track, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame3, text="Mono>=").pack(side=tk.LEFT, padx=(8, 0))
        mono_entry = ttk.Entry(frame3, textvariable=self.regex_track_mono, width=5)
        mono_entry.pack(side=tk.LEFT, padx=2)
        mono_entry.bind("<Return>", lambda e: self.update())

    def _on_regex_change(self, event=None):
        pattern = self.regex_pattern.get()
        try:
            re.compile(pattern)
            self.regex_entry.configure(foreground="black")
        except:
            self.regex_entry.configure(foreground="red")

    def _set_pattern(self, pattern):
        self.regex_pattern.set(pattern)
        self.update()

    def _get_files_data(self):
        from sparams_io import get_cached_network, display_name
        fmin, fmax, sstr = self.get_freq_range()
        param = self.regex_param.get()
        pattern = self.regex_pattern.get()

        files_data = []
        all_files_info = []

        for v, p, d in self.get_files():
            fname = display_name(p, d)
            value = extract_regex_value(fname, pattern, self.regex_group.get())

            if v.get():
                all_files_info.append(f"{'Y' if value is not None else 'N'} {fname} -> "
                                      f"{value if value is not None else 'no match'}")

            if not v.get() or value is None:
                continue

            ntw = get_cached_network(p, d, sstr)
            if ntw is None:
                continue

            try:
                if self.regex_gate.get():
                    raw_data = getattr(ntw, param).time_gate(
                        center=self.regex_gate_center.get(),
                        span=self.regex_gate_span.get())
                else:
                    raw_data = getattr(ntw, param)

                if self.regex_phase.get():
                    phase_deg = np.degrees(np.unwrap(np.angle(raw_data.s.flatten())))
                    s_data = (phase_deg + 180) % 360 - 180
                else:
                    s_data = raw_data.s_db.flatten() if self.get_scale_mode() else raw_data.s_mag.flatten()

                files_data.append((fname, value, ntw.f, s_data, d))

            except Exception as e:
                print(f"regex: {fname} skipped: {e}", file=sys.stderr)

        self.all_files_info = all_files_info
        return files_data

    def update(self):
        self.fig.clf()
        self.ax_side = None
        if self.regex_track.get():
            gs = self.fig.add_gridspec(1, 4, wspace=0.32)
            ax = self.fig.add_subplot(gs[0, :3])
            self.ax_side = self.fig.add_subplot(gs[0, 3])
        else:
            ax = self.fig.add_subplot(111)

        files_data = self._get_files_data()

        if not files_data:
            info = getattr(self, "all_files_info", [])
            n_match = sum(1 for s in info if s.startswith("Y"))
            if not info:
                msg = "No files selected"
            elif n_match == 0:
                msg = f"{len(info)} selected, none match pattern: {self.regex_pattern.get()}"
            else:
                msg = f"{n_match} matched but none could be loaded for {self.get_freq_range()[2]}\n(see terminal for skip reasons)"
            ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            self._update_legend_panel([])
            self.last_result = None
            return

        files_data.sort(key=lambda x: x[1])
        # files may have different sweeps; the per-frequency analysis needs one shared grid,
        # so resample every curve onto the densest one (a no-op when they already match)
        freqs = max((d[2] for d in files_data), key=len)
        s_matrix = np.array([np.interp(freqs, d[2], d[3]) for d in files_data])
        labels = [d[0] for d in files_data]
        n_pts = len(freqs)

        n_drop = min(self.regex_n_drop.get(), len(files_data) - 2)
        mono_mask, dropped_idx = find_best_drop(s_matrix, max(n_drop, 0), self.regex_strict.get())
        dropped_labels = [labels[i] for i in dropped_idx]
        mono_ranges = mask_to_ranges(mono_mask, freqs) if self.regex_highlight.get() else []

        tau_ranges = None
        tau_mask = None
        if self.regex_tau.get():
            taus = compute_kendall_tau(s_matrix)
            tau_mask = np.abs(taus) >= self.regex_tau_threshold.get()
            tau_ranges = mask_to_ranges(tau_mask, freqs)

        disp_ranges = None
        disp_mask = None
        if self.regex_disp.get():
            disps = compute_max_displacement(s_matrix)
            disp_mask = disps <= self.regex_disp_threshold.get()
            disp_ranges = mask_to_ranges(disp_mask, freqs)

        congru_ranges = None
        congru_iranges = None
        congru_mask = None
        if self.regex_congru.get() and s_matrix.shape[0] >= 2:
            mode = self.regex_congru_mode.get()
            _, congru_mask = compute_congruence(
                s_matrix,
                window=self.regex_congru_window.get(),
                threshold=self.regex_congru_threshold.get(),
                mode=mode,
                max_lag=self.regex_congru_maxlag.get())
            congru_iranges = mask_to_index_ranges(congru_mask)
            congru_ranges = [(freqs[a], freqs[b]) for a, b in congru_iranges]

        track_mask = None
        track_region = None
        if self.regex_track.get() and s_matrix.shape[0] >= 3:
            values = np.array([d[1] for d in files_data])
            max_lag = self.regex_congru_maxlag.get()
            # the analysis window must comfortably exceed the full shift span,
            # so size it from Max shift rather than the (small) congruence window
            track_window = max(self.regex_congru_window.get(), 3 * max_lag + 1)
            track_mask, track_region = compute_shift_tracking(
                s_matrix, values, freqs,
                window=track_window,
                max_lag=max_lag,
                shape_thresh=self.regex_congru_threshold.get(),
                mono_thresh=self.regex_track_mono.get())

        dropped_set = set(dropped_idx) if n_drop > 0 else set()
        legend_items = []
        for i, (fname, value, freq, s_data, file_dict) in enumerate(files_data):
            is_dropped = i in dropped_set
            kwargs = {'label': fname}
            if file_dict.get('line_color'):
                kwargs['color'] = file_dict['line_color']
            else:
                cmap = plt.cm.viridis(np.linspace(0, 1, len(files_data)))
                kwargs['color'] = cmap[i]
            if file_dict.get('line_width', 1.0) != 1.0:
                kwargs['linewidth'] = file_dict['line_width']
            if is_dropped:
                kwargs['alpha'] = 0.3
                kwargs['linestyle'] = '--'
            line = ax.plot(freq, s_data, **kwargs)[0]
            legend_items.append((fname, matplotlib.colors.to_hex(line.get_color()), is_dropped))

        if self.regex_highlight.get():
            for f1, f2 in mono_ranges:
                ax.axvspan(f1, f2, color='green', alpha=0.3)
            if self.regex_small_diff.get():
                small_mask = compute_small_diffs(s_matrix, freqs, mono_mask,
                    self.regex_small_threshold.get(), self.regex_small_ref.get(),
                    self.regex_small_count.get())
                small_ranges = mask_to_ranges(small_mask, freqs)
                for f1, f2 in small_ranges:
                    ax.axvspan(f1, f2, color='blue', alpha=0.5)

        if tau_ranges:
            for f1, f2 in tau_ranges:
                ax.axvspan(f1, f2, color='orange', alpha=0.2)

        if disp_ranges:
            for f1, f2 in disp_ranges:
                ax.axvspan(f1, f2, color='purple', alpha=0.2)

        if congru_iranges:
            mode = self.regex_congru_mode.get()
            max_lag = self.regex_congru_maxlag.get()
            for a, b in congru_iranges:
                ax.axvspan(freqs[a], freqs[b], color='magenta', alpha=0.15)
                vals, off = congruence_template(s_matrix[:, a:b + 1], mode, max_lag)
                fseg = freqs[a + off: a + off + len(vals)]
                ax.plot(fseg, vals, color='black', linestyle='--', linewidth=2.0, zorder=5)

        if track_mask is not None:
            for a, b in mask_to_index_ranges(track_mask):
                ax.axvspan(freqs[a], freqs[b], color='cyan', alpha=0.2)
            self._draw_shift_side(track_region)

        ax.set_xlabel("Frequency [Hz]")
        if self.regex_phase.get():
            ylabel = f"Phase {self.regex_param.get().upper()} [deg]"
        else:
            ylabel = f"|{self.regex_param.get().upper()}| [dB]"
        ax.set_ylabel(ylabel)

        title = f"Regex ordering: {self.regex_pattern.get()} (group {self.regex_group.get()})"
        if n_drop > 0:
            title += f" drop {n_drop}"
        if self.regex_phase.get():
            title += " Phase"
        if self.regex_gate.get():
            title += f" [Gated: {self.regex_gate_center.get()}+/-{self.regex_gate_span.get()/2}ns]"
        ax.set_title(title)
        ax.grid(True)

        if self.get_legend_on_plot() and legend_items:
            handles, lbls = ax.get_legend_handles_labels()
            if lbls:
                ncol = min(5, (len(lbls) - 1) // 10 + 1)
                ax.legend(handles, lbls, loc='best', ncol=ncol, fontsize=8, framealpha=0.9)

        self.fig.tight_layout()
        self.canvas.draw()

        self._update_legend_panel(legend_items)

        self.regex_ranges = mono_ranges
        self.last_result = {
            'pattern': self.regex_pattern.get(),
            'group': self.regex_group.get(),
            'files_info': self.all_files_info,
            'mono_ranges': mono_ranges,
            'mono_coverage': 100.0 * mono_mask.sum() / n_pts if n_pts > 0 else 0,
            'dropped_labels': dropped_labels,
            'n_freq_points': n_pts,
        }
        if tau_ranges is not None:
            self.last_result['tau_ranges'] = tau_ranges
            self.last_result['tau_threshold'] = self.regex_tau_threshold.get()
            self.last_result['tau_coverage'] = 100.0 * tau_mask.sum() / n_pts
        if disp_ranges is not None:
            self.last_result['disp_ranges'] = disp_ranges
            self.last_result['disp_threshold'] = self.regex_disp_threshold.get()
            self.last_result['disp_coverage'] = 100.0 * disp_mask.sum() / n_pts
        if congru_ranges is not None:
            self.last_result['congru_ranges'] = congru_ranges
            self.last_result['congru_threshold'] = self.regex_congru_threshold.get()
            self.last_result['congru_mode'] = self.regex_congru_mode.get()
            self.last_result['congru_coverage'] = 100.0 * congru_mask.sum() / n_pts
        if track_region is not None:
            self.last_result['track'] = track_region

    def _draw_shift_side(self, region):
        ax = self.ax_side
        if ax is None:
            return
        if region is None:
            ax.text(0.5, 0.5, "no coherent\nsliding region",
                    ha="center", va="center", color="gray", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        v = region['values']
        sh = region['shifts_hz'] / 1e6
        ax.plot(v, sh, 'o', color='crimson')
        slope = region['slope_hz_per_unit'] / 1e6
        icpt = region['intercept'] / 1e6
        xs = np.array([v.min(), v.max()])
        ax.plot(xs, slope * xs + icpt, '--', color='gray')
        ax.set_title(f"{slope:.1f} MHz/unit\nR^2={region['r2']:.2f}  "
                     f"@ {region['f0']/1e9:.2f}-{region['f1']/1e9:.2f} GHz", fontsize=8)
        ax.set_xlabel("regex value", fontsize=8)
        ax.set_ylabel("feature shift [MHz]", fontsize=8)
        ax.grid(True)

    def _update_legend_panel(self, legend_items):
        for widget in self.legend_frame.winfo_children():
            widget.destroy()

        if not legend_items:
            ttk.Label(self.legend_frame, text="No data",
                      foreground="gray").pack(pady=10)
            return

        for entry in legend_items:
            name, color = entry[0], entry[1]
            is_dropped = entry[2] if len(entry) > 2 else False
            row = ttk.Frame(self.legend_frame)
            row.pack(fill=tk.X, padx=5, pady=2)

            color_label = tk.Label(row, text="\u25a0 ", foreground=color, font=("", 12))
            color_label.pack(side=tk.LEFT, padx=(0, 5))

            display_name = name if len(name) <= 20 else name[:17] + "..."
            if is_dropped:
                display_name += " (dropped)"
            ttk.Label(row, text=display_name, font=("", 9),
                      foreground="gray" if is_dropped else "").pack(side=tk.LEFT)

        self.legend_canvas.configure(scrollregion=self.legend_canvas.bbox("all"))

    def get_text_output(self):
        if self.last_result is None:
            return ""
        fmin, fmax, _ = self.get_freq_range()
        return format_regex_text(
            self.last_result['pattern'],
            self.last_result['group'],
            (fmin, fmax),
            self.last_result
        )

    def _export_ranges(self):
        all_ranges = []
        if self.regex_ranges:
            all_ranges += [("monotonic", r) for r in self.regex_ranges]
        if self.last_result and 'tau_ranges' in self.last_result:
            all_ranges += [("kendall_tau", r) for r in self.last_result['tau_ranges']]
        if self.last_result and 'disp_ranges' in self.last_result:
            all_ranges += [("max_disp", r) for r in self.last_result['disp_ranges']]
        if self.last_result and 'congru_ranges' in self.last_result:
            mode = self.last_result.get('congru_mode', 'vertical')
            all_ranges += [(f"shared_shape_{mode}", r) for r in self.last_result['congru_ranges']]

        if not all_ranges:
            messagebox.showinfo("No ranges", "No frequency ranges found.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save frequency ranges",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            with open(filename, 'w') as f:
                f.write(f"# Frequency ranges for pattern: {self.regex_pattern.get()}\n")
                f.write(f"# S-parameter: {self.regex_param.get().upper()}\n")
                mode = "Strict" if self.regex_strict.get() else "Non-strict"
                f.write(f"# Monotonicity: {mode}, Drop: {self.regex_n_drop.get()}\n")
                if self.regex_gate.get():
                    f.write(f"# Time gating: Center={self.regex_gate_center.get()}ns, Span={self.regex_gate_span.get()}ns\n")
                f.write(f"# Format: type start_freq end_freq (Hz)\n")
                f.write("#\n")
                for label, (start, end) in all_ranges:
                    f.write(f"{label} {start} {end}\n")

            messagebox.showinfo("Export complete", f"Ranges saved to {filename}")

    def _open_stats(self):
        self._stats_window(grouped=False)

    def _open_stats_group(self):
        self._stats_window(grouped=True)

    def _stats_collect(self):
        # selected files matching the pattern, each with its full (unsliced) network loaded
        from sparams_io import get_cached_network, display_name
        sstr = self.get_freq_range()[2]
        out = []
        for v, p, d in self.get_files():
            if not v.get():
                continue
            name = display_name(p, d)
            val = extract_regex_value(name, self.regex_pattern.get(), self.regex_group.get())
            if val is None:
                continue
            if get_cached_network(p, d, sstr) is None:   # side effect: loads d['ntwk_full']
                continue
            out.append((name, val, d['ntwk_full']))
        return out

    def _stats_recalc(self, txt, mode_var, fmin_var, fmax_var, single_var, grouped):
        param = self.regex_param.get()
        pattern, gi = self.regex_pattern.get(), self.regex_group.get()
        files = self._stats_collect()
        report = "No files selected / matching pattern."
        if files and mode_var.get() == "range":
            fmin, fmax = fmin_var.get(), fmax_var.get()
            sstr = f"{fmin}-{fmax}ghz"
            curves = []
            for name, val, full in files:
                try:
                    ntw = full[sstr]
                    curves.append((name, val, ntw.f, getattr(ntw, param).s_db.flatten()))
                except Exception as e:
                    print(f"stats: {name} skipped: {e}", file=sys.stderr)
            if curves:
                curves.sort(key=lambda x: x[1])
                ref = max((c[2] for c in curves), key=len)
                curves = [(n, v, ref, np.interp(ref, f, s)) for n, v, f, s in curves]
                if grouped:
                    groups = {}
                    for n, v, f, s in curves:
                        groups.setdefault(v, []).append((n, f, s))
                    report = format_group_range(groups, fmin, fmax, pattern, gi)
                else:
                    report = format_perfile_range([(n, f, s) for n, v, f, s in curves], fmin, fmax)
        elif files:
            single = single_var.get()
            vals = []
            for name, val, full in files:
                try:
                    f = full.f
                    s = getattr(full, param).s_db.flatten()
                    i = int(np.argmin(np.abs(f - single * 1e9)))
                    vals.append((name, val, float(s[i])))
                except Exception as e:
                    print(f"stats: {name} skipped: {e}", file=sys.stderr)
            if vals:
                vals.sort(key=lambda x: x[1])
                if grouped:
                    groups = {}
                    for n, v, vv in vals:
                        groups.setdefault(v, []).append((n, vv))
                    report = format_group_single(groups, single, pattern, gi)
                else:
                    report = format_perfile_single([(n, vv) for n, v, vv in vals], single)
        txt.config(state=tk.NORMAL)
        txt.delete("1.0", tk.END)
        txt.insert(tk.END, report)
        txt.config(state=tk.DISABLED)

    def _stats_window(self, grouped):
        win = tk.Toplevel(self.parent)
        win.title("Regex Stats Group" if grouped else "Regex Stats")
        win.geometry("760x620")

        fmin0, fmax0, _ = self.get_freq_range()
        mode_var = tk.StringVar(value="range")
        fmin_var = tk.DoubleVar(value=fmin0)
        fmax_var = tk.DoubleVar(value=fmax0)
        single_var = tk.DoubleVar(value=round((fmin0 + fmax0) / 2, 4))

        top = ttk.Frame(win)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Label(top, text="Analysis:").pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Range", value="range", variable=mode_var).pack(side=tk.LEFT)
        ttk.Label(top, text="fmin[GHz]").pack(side=tk.LEFT, padx=(6, 1))
        ttk.Entry(top, textvariable=fmin_var, width=7).pack(side=tk.LEFT)
        ttk.Label(top, text="fmax[GHz]").pack(side=tk.LEFT, padx=(6, 1))
        ttk.Entry(top, textvariable=fmax_var, width=7).pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Single", value="single", variable=mode_var).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(top, text="f[GHz]").pack(side=tk.LEFT, padx=(6, 1))
        ttk.Entry(top, textvariable=single_var, width=7).pack(side=tk.LEFT)

        body = ttk.Frame(win)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(body)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(body, wrap="none", font=("Consolas", 9), yscrollcommand=scroll.set)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=txt.yview)

        def recalc():
            self._stats_recalc(txt, mode_var, fmin_var, fmax_var, single_var, grouped)
        ttk.Button(top, text="Recalc", command=recalc).pack(side=tk.LEFT, padx=10)

        bottom = ttk.Frame(win)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)
        ttk.Button(bottom, text="Copy all",
                   command=lambda: (win.clipboard_clear(), win.clipboard_append(txt.get("1.0", tk.END)))).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom, text="Save txt", command=lambda: self._stats_save(txt)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom, text="Export csv",
                   command=lambda: self._stats_csv(mode_var, fmin_var, fmax_var, single_var)).pack(side=tk.LEFT, padx=2)

        from ui_util import bind_enter
        bind_enter(top, recalc)
        recalc()

    def _stats_save(self, txt):
        fn = filedialog.asksaveasfilename(defaultextension=".txt",
                                          filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if fn:
            with open(fn, "w") as f:
                f.write(txt.get("1.0", tk.END))

    def _stats_csv(self, mode_var, fmin_var, fmax_var, single_var):
        import csv
        files = self._stats_collect()
        if not files:
            messagebox.showinfo("No data", "Nothing to export")
            return
        fn = filedialog.asksaveasfilename(defaultextension=".csv",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not fn:
            return
        param = self.regex_param.get()
        rows = []
        if mode_var.get() == "range":
            sstr = f"{fmin_var.get()}-{fmax_var.get()}ghz"
            for name, val, full in files:
                try:
                    ntw = full[sstr]
                    s = getattr(ntw, param).s_db.flatten()
                    f = ntw.f
                except Exception:
                    continue
                i = int(np.argmin(s))
                rows.append([name, val, float(np.mean(s)), f[i] / 1e9, float(s[i])])
            header = ["file", "regex_value", "mean_dB", "f_min_GHz", "min_dB"]
        else:
            sg = single_var.get()
            for name, val, full in files:
                f = full.f
                s = getattr(full, param).s_db.flatten()
                i = int(np.argmin(np.abs(f - sg * 1e9)))
                rows.append([name, val, float(s[i])])
            header = ["file", "regex_value", f"value_dB_at_{sg:g}GHz"]
        rows.sort(key=lambda r: r[1])
        with open(fn, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(header)
            w.writerows(rows)
        messagebox.showinfo("Export complete", f"Saved to {fn}")
