import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from pathlib import Path
import re

from analysis_regex import extract_regex_value, analyze_ordered_ranges, format_regex_text

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
        self.regex_small_diff = tk.BooleanVar(value=False)
        self.regex_small_threshold = tk.DoubleVar(value=0.1)
        self.regex_small_ref = tk.DoubleVar(value=10.0)
        self.regex_small_count = tk.IntVar(value=1)
        self.regex_gate = tk.BooleanVar(value=False)
        self.regex_gate_center = tk.DoubleVar(value=5.0)
        self.regex_gate_span = tk.DoubleVar(value=0.5)
        self.regex_phase = tk.BooleanVar(value=False)
        
        self.regex_ranges = []
        self.regex_spans = []
        self.last_result = None
        
        self._build_ui()
    
    def _build_ui(self):
        
        frame = ttk.Frame(self.control_frame)
        frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        ttk.Label(frame, text="Pattern:").pack(side=tk.LEFT, padx=(0,5))
        self.regex_entry = ttk.Entry(frame, textvariable=self.regex_pattern, width=20)
        self.regex_entry.pack(side=tk.LEFT, padx=2)
        self.regex_entry.bind("<KeyRelease>", self._on_regex_change)
        self.regex_entry.bind("<Return>", lambda e: self.update())
        
        ttk.Label(frame, text="Group:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Spinbox(frame, from_=1, to=10, width=3,
                   textvariable=self.regex_group, command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(frame, text="Param:").pack(side=tk.LEFT, padx=(0,5))
        for n in ("s11", "s12", "s21", "s22"):
            ttk.Radiobutton(frame, text=n.upper(), value=n,
                          variable=self.regex_param, command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Checkbutton(frame, text="Gating", variable=self.regex_gate,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame, text="Center[ns]").pack(side=tk.LEFT, padx=(5,2))
        gate_center_entry = ttk.Entry(frame, textvariable=self.regex_gate_center, width=5)
        gate_center_entry.pack(side=tk.LEFT, padx=2)
        gate_center_entry.bind("<Return>", lambda e: self.update())
        ttk.Label(frame, text="Span[ns]").pack(side=tk.LEFT, padx=(5,2))
        gate_span_entry = ttk.Entry(frame, textvariable=self.regex_gate_span, width=5)
        gate_span_entry.pack(side=tk.LEFT, padx=2)
        gate_span_entry.bind("<Return>", lambda e: self.update())
        
        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Checkbutton(frame, text="Highlight monotonic",
                       variable=self.regex_highlight, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(frame, text="Strict",
                       variable=self.regex_strict, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(frame, text="Phase",
                       variable=self.regex_phase, command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(frame, text="Export", command=self._export_ranges).pack(side=tk.LEFT, padx=10)
        
        frame2 = ttk.Frame(self.control_frame)
        frame2.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        ttk.Checkbutton(frame2, text="Small diffs",
                       variable=self.regex_small_diff, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame2, text="Tol:").pack(side=tk.LEFT, padx=(5,2))
        small_tol_entry = ttk.Entry(frame2, textvariable=self.regex_small_threshold, width=5)
        small_tol_entry.pack(side=tk.LEFT, padx=2)
        small_tol_entry.bind("<Return>", lambda e: self.update())
        ttk.Label(frame2, text="@").pack(side=tk.LEFT, padx=2)
        small_ref_entry = ttk.Entry(frame2, textvariable=self.regex_small_ref, width=5)
        small_ref_entry.pack(side=tk.LEFT, padx=2)
        small_ref_entry.bind("<Return>", lambda e: self.update())
        ttk.Label(frame2, text="dB").pack(side=tk.LEFT, padx=(0,10))
        
        ttk.Separator(frame2, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(frame2, text="Quick:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(frame2, text="_XXml", width=6,
                  command=lambda: self._set_pattern(r'_(\d+)ml')).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame2, text="XXmm", width=6,
                  command=lambda: self._set_pattern(r'(\d+)mm')).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame2, text="XXcm", width=6,
                  command=lambda: self._set_pattern(r'(\d+)cm')).pack(side=tk.LEFT, padx=2)
        
        

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
        fmin, fmax, sstr = self.get_freq_range()
        param = self.regex_param.get()
        pattern = self.regex_pattern.get()
        
        files_data = []
        all_files_info = []
        
        for v, p, d in self.get_files():
            fname = d.get('custom_name') if d.get('is_average') else Path(p).stem
            value = extract_regex_value(fname, pattern, self.regex_group.get())
            
            if v.get():
                all_files_info.append(f"{'Y' if value is not None else 'N'} {fname} -> "
                                    f"{value if value is not None else 'no match'}")
            
            if not v.get() or value is None:
                continue
            ext = Path(p).suffix.lower()
            is_touchstone = re.match(r"\.s\d+p$", ext) is not None

            if d.get('is_average', False) or is_touchstone or ext == ".csv":
                try:
                    from sparams_io import loadFile
                    
                    ntw_full = d.get('ntwk_full')
                    if ntw_full is None:
                        ntw_full = loadFile(p)
                        d['ntwk_full'] = ntw_full
                    
                    ntw = ntw_full[sstr]
                    
                    if self.regex_gate.get():
                        raw_param = getattr(ntw, param)
                        gated_param = raw_param.time_gate(
                            center=self.regex_gate_center.get(),
                            span=self.regex_gate_span.get()
                        )
                        raw_data = gated_param
                    else:
                        raw_data = getattr(ntw, param)
                    
                    freq = ntw.f
                    
                    if self.regex_phase.get():
                        phase_rad = np.unwrap(np.angle(raw_data.s.flatten()))
                        phase_deg = np.degrees(phase_rad)
                        s_data = (phase_deg + 180) % 360 - 180
                    else:
                        use_db = self.get_scale_mode()
                        s_data = raw_data.s_db.flatten() if use_db else raw_data.s_mag.flatten()
                    
                    files_data.append((fname, value, freq, s_data, d))
                    
                except Exception:
                    pass
        
        self.all_files_info = all_files_info
        return files_data
    
    def update(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        
        files_data = self._get_files_data()
        
        if not files_data:
            ax.text(0.5, 0.5, f"No files match pattern: {self.regex_pattern.get()}\nCheck regex and capture group",
                   ha="center", va="center", fontsize=12, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            self._update_legend_panel([])
            self.last_result = None
            return
        
        analysis_data = [(f, v, fr, sd) for f, v, fr, sd, _ in files_data]
        ranges, freqs, s_matrix, labels, small_diff_ranges = analyze_ordered_ranges(
            analysis_data,
            self.regex_strict.get(),
            self.regex_small_threshold.get(),
            self.regex_small_ref.get(),
            self.regex_small_count.get()
        )
        
        self.regex_ranges = ranges
        
        files_data.sort(key=lambda x: x[1])
        
        legend_items = []
        for i, (fname, value, freq, s_data, file_dict) in enumerate(files_data):
            kwargs = {'label': fname}
            if file_dict.get('line_color'):
                kwargs['color'] = file_dict['line_color']
            else:
                cmap = plt.cm.viridis(np.linspace(0, 1, len(files_data)))
                kwargs['color'] = cmap[i]
            if file_dict.get('line_width', 1.0) != 1.0:
                kwargs['linewidth'] = file_dict['line_width']
            
            line = ax.plot(freq, s_data, **kwargs)[0]
            legend_items.append((fname, matplotlib.colors.to_hex(line.get_color())))
        
        for span in self.regex_spans:
            try:
                span.remove()
            except:
                pass
        self.regex_spans.clear()
        
        if self.regex_highlight.get():
            for (f1, f2) in ranges:
                span = ax.axvspan(f1, f2, color='green', alpha=0.3)
                self.regex_spans.append(span)
            
            if self.regex_small_diff.get():
                for (f1, f2) in small_diff_ranges:
                    span = ax.axvspan(f1, f2, color='blue', alpha=0.5)
                    self.regex_spans.append(span)
        
        ax.set_xlabel("Frequency [Hz]")
        ylabel = f"Phase {self.regex_param.get().upper()} [Ã‚Â°]" if self.regex_phase.get() else f"|{self.regex_param.get().upper()}| [dB]"
        ax.set_ylabel(ylabel)
        
        title = f"Regex-based ordering: {self.regex_pattern.get()} (group {self.regex_group.get()})"
        if self.regex_phase.get():
            title += " Ã¢â‚¬â€ Phase"
        if self.regex_gate.get():
            title += f" [Gated: {self.regex_gate_center.get()}Ã‚Â±{self.regex_gate_span.get()/2}ns]"
        ax.set_title(title)
        ax.grid(True)
        
        if self.get_legend_on_plot() and legend_items:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ncol = min(5, (len(labels) - 1) // 10 + 1)
                ax.legend(handles, labels, loc='best', ncol=ncol, fontsize=8, framealpha=0.9)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        self._update_legend_panel(legend_items)
        
        self.last_result = {
            'pattern': self.regex_pattern.get(),
            'group': self.regex_group.get(),
            'ranges': ranges,
            'files_info': self.all_files_info
        }
    
    def _update_legend_panel(self, legend_items):
        for widget in self.legend_frame.winfo_children():
            widget.destroy()
        
        if not legend_items:
            ttk.Label(self.legend_frame, text="No data",
                     foreground="gray").pack(pady=10)
            return
        
        for name, color in legend_items:
            row = ttk.Frame(self.legend_frame)
            row.pack(fill=tk.X, padx=5, pady=2)
            
            color_label = tk.Label(row, text="\u25a0 ", foreground=color, font=("", 12))
            color_label.pack(side=tk.LEFT, padx=(0, 5))
            
            display_name = name if len(name) <= 20 else name[:17] + "..."
            ttk.Label(row, text=display_name, font=("", 9)).pack(side=tk.LEFT)
        
        self.legend_canvas.configure(scrollregion=self.legend_canvas.bbox("all"))
    
    def get_text_output(self):
        if self.last_result is None:
            return ""
        
        fmin, fmax, _ = self.get_freq_range()
        return format_regex_text(
            self.last_result['pattern'],
            self.last_result['group'],
            (fmin, fmax),
            self.last_result['ranges'],
            self.last_result['files_info']
        )
    
    def _export_ranges(self):
        if not self.regex_ranges:
            messagebox.showinfo("No ranges", "No monotonic frequency ranges found.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save frequency ranges",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(f"# Monotonic frequency ranges for pattern: {self.regex_pattern.get()}\n")
                f.write(f"# S-parameter: {self.regex_param.get().upper()}\n")
                mode = "Strict monotonic" if self.regex_strict.get() else "Non-strict monotonic"
                f.write(f"# Monotonicity mode: {mode}\n")
                if self.regex_gate.get():
                    f.write(f"# Time gating: Center={self.regex_gate_center.get()}ns, Span={self.regex_gate_span.get()}ns\n")
                f.write(f"# Format: start_freq end_freq (Hz)\n")
                f.write("#\n")
                for start, end in self.regex_ranges:
                    f.write(f"{start} {end}\n")
            
            messagebox.showinfo("Export complete", f"Ranges saved to {filename}")
