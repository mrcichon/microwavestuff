import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from pathlib import Path

from analysis_shape import compute_shape_matrix, format_shape_text

class TabShapeComparison:
    def __init__(self, parent, fig, canvas,
                 get_files_func, get_freq_range_func, get_scale_mode_func):
        self.parent = parent
        self.fig = fig
        self.canvas = canvas
        self.ax = None
        self.ax_activity = None
        self.get_files = get_files_func
        self.get_freq_range = get_freq_range_func
        self.get_scale_mode = get_scale_mode_func
        
        self.shape_param = tk.StringVar(value="s21")
        self.shape_metric = tk.StringVar(value="xcorr")
        self.shape_normalize = tk.BooleanVar(value=True)
        self.shape_adaptive = tk.BooleanVar(value=False)
        self.shape_alpha = tk.DoubleVar(value=2.0)
        self.shape_sigma = tk.IntVar(value=5)
        self.shape_gamma = tk.DoubleVar(value=2.0)
        self.shape_wmin = tk.DoubleVar(value=0.1)
        self.shape_activity_type = tk.StringVar(value="nonincreasing")
        self.shape_show_activity = tk.BooleanVar(value=False)
        self.max_lag = tk.IntVar(value=5)
        
        self.shape_data = None
        
        self._build_ui()
    
    def _build_ui(self):
        # Main control bar
        frame = ttk.Frame(self.parent)
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame, text="Param:").pack(side=tk.LEFT, padx=(0,5))
        for n in ("s11", "s12", "s21", "s22"):
            ttk.Radiobutton(frame, text=n.upper(), value=n,
                          variable=self.shape_param, command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Checkbutton(frame, text="Normalize", variable=self.shape_normalize,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(frame, text="Metric:").pack(side=tk.LEFT, padx=(0,5))
        for text, value in [("XCorr", "xcorr"), ("L1", "l1"), ("L2", "l2"),
                           ("AL1", "al1"), ("AL2", "al2")]:
            ttk.Radiobutton(frame, text=text, value=value,
                          variable=self.shape_metric, command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(frame, text="MaxLag:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Spinbox(frame, from_=1, to=1000, textvariable=self.max_lag,
                   width=6, command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(frame, text="Export", command=self._export_matrix).pack(side=tk.LEFT, padx=10)
        
        # Adaptive parameters on second row (shown/hidden dynamically)
        self.adapt_frame = ttk.Frame(self.parent)
        self.adapt_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)
        self.adapt_frame.pack_forget()  # Initially hidden
        
        ttk.Label(self.adapt_frame, text="Adaptive:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Label(self.adapt_frame, text="Î±:").pack(side=tk.LEFT)
        ttk.Spinbox(self.adapt_frame, from_=1.0, to=4.0, increment=0.5,
                   textvariable=self.shape_alpha, width=5,
                   command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(self.adapt_frame, text="Ïƒ:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Spinbox(self.adapt_frame, from_=2, to=20,
                   textvariable=self.shape_sigma, width=5,
                   command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(self.adapt_frame, text="Î³:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Spinbox(self.adapt_frame, from_=0.5, to=4.0, increment=0.5,
                   textvariable=self.shape_gamma, width=5,
                   command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(self.adapt_frame, text="w_min:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Spinbox(self.adapt_frame, from_=0.0, to=1.0, increment=0.1,
                   textvariable=self.shape_wmin, width=5,
                   command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Radiobutton(self.adapt_frame, text="Bidirectional", value="bidirectional",
                       variable=self.shape_activity_type,
                       command=self.update).pack(side=tk.LEFT, padx=(10,2))
        ttk.Radiobutton(self.adapt_frame, text="Non-increasing", value="nonincreasing",
                       variable=self.shape_activity_type,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.adapt_frame, text="Show activity",
                       variable=self.shape_show_activity,
                       command=self.update).pack(side=tk.LEFT, padx=(10,2))
        
        self.control_panel = frame



    def _get_files_data(self):
        fmin, fmax, sstr = self.get_freq_range()
        param = self.shape_param.get()
        
        files_data = []
        for v, p, d in self.get_files():
            if not v.get():
                continue
            
            ext = Path(p).suffix.lower()
            if d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']:
                try:
                    from sparams_io import loadFile
                    
                    ntw_full = d.get('ntwk_full')
                    if ntw_full is None:
                        ntw_full = loadFile(p)
                        d['ntwk_full'] = ntw_full
                    
                    cached_range = d.get('cached_range')
                    if cached_range != sstr:
                        ntw = ntw_full[sstr]
                        d['ntwk'] = ntw
                        d['cached_range'] = sstr
                    else:
                        ntw = d['ntwk']
                    
                    fname = d.get('custom_name') if d.get('is_average') else Path(p).stem
                    
                    use_db = self.get_scale_mode()
                    s_param = getattr(ntw, param)
                    s_data = s_param.s_db.flatten() if use_db else s_param.s_mag.flatten()
                    
                    files_data.append({
                        'name': fname,
                        'signal': s_data,
                        'freq': ntw.f
                    })
                    
                except Exception:
                    pass
        
        return files_data
    
    def update(self):
        # Show/hide adaptive frame
        is_adaptive = self.shape_metric.get() in ["al1", "al2"]
        if is_adaptive:
            self.adapt_frame.pack(anchor="w", pady=(5,0), fill="x")
        else:
            self.adapt_frame.pack_forget()
        
        # Clear figure and create subplots
        self.fig.clear()
        if self.shape_show_activity.get() and is_adaptive:
            self.ax = self.fig.add_subplot(211)
            self.ax_activity = self.fig.add_subplot(212)
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax_activity = None
        
        files_data = self._get_files_data()
        
        if len(files_data) < 2:
            self.ax.text(0.5, 0.5, "Need at least 2 files selected",
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            self.shape_data = None
            return
        
        # Prepare adaptive params if needed
        adaptive_params = None
        if is_adaptive:
            adaptive_params = {
                'alpha': self.shape_alpha.get(),
                'sigma': self.shape_sigma.get(),
                'gamma': self.shape_gamma.get(),
                'w_min': self.shape_wmin.get(),
                'activity_type': self.shape_activity_type.get()
            }
        
        # Compute shape matrix
        max_lag_val = self.max_lag.get() if self.max_lag.get() > 0 else None
        self.shape_data = compute_shape_matrix(
            files_data,
            self.shape_param.get(),
            self.shape_metric.get(),
            self.shape_normalize.get(),
            max_lag_val,
            adaptive_params
        )
        
        if self.shape_data is None:
            self.ax.text(0.5, 0.5, "Error computing shape matrix",
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            return
        
        # Plot activity if requested
        if self.ax_activity and self.shape_data['activity'] is not None:
            freqs = self.shape_data['frequencies']
            activity = self.shape_data['activity']
            weights = self.shape_data['weights']
            
            self.ax_activity.plot(freqs, activity, 'r-', label='Activity', linewidth=2)
            if weights is not None:
                weights_scaled = weights * np.max(activity) / np.max(weights)
                self.ax_activity.plot(freqs, weights_scaled, 'g-', label='Weights (scaled)', alpha=0.7)
            
            self.ax_activity.set_xlabel('Frequency [Hz]')
            self.ax_activity.set_ylabel('Activity / Weight')
            self.ax_activity.legend()
            self.ax_activity.grid(True, alpha=0.3)
        
        # Plot matrix
        matrix = self.shape_data['matrix']
        n = matrix.shape[0]
        names = self.shape_data['names']
        
        if self.shape_metric.get() == "xcorr":
            im = self.ax.imshow(matrix, cmap='RdBu_r', aspect='auto',
                              interpolation='nearest', vmin=-1, vmax=1)
        else:
            vmax = np.percentile(matrix[np.triu_indices(n, 1)], 95) if n > 1 else 1
            im = self.ax.imshow(matrix, cmap='RdBu_r', aspect='auto',
                              interpolation='nearest', vmin=0, vmax=vmax if vmax > 0 else None)
        
        self.ax.set_xticks(range(n))
        self.ax.set_yticks(range(n))
        self.ax.set_xticklabels(names, rotation=45, ha='right')
        self.ax.set_yticklabels(names)
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                if self.shape_metric.get() == "xcorr":
                    tc = "white" if abs(matrix[i, j]) > 0.5 else "black"
                    txt = f'{matrix[i, j]:.2f}'
                    if i != j:
                        lag = self.shape_data['lag_matrix'][i, j]
                        txt += f'\n({int(lag)})'
                else:
                    tc = "white" if (matrix[i, j] > (np.max(matrix) * 0.6 if np.max(matrix) > 0 else 0)) else "black"
                    txt = f'{matrix[i, j]:.3f}'
                
                self.ax.text(j, i, txt, ha="center", va="center", color=tc, fontsize=7)
        
        metric_names = {"xcorr": "Cross-Correlation", "l1": "L1 Distance", "l2": "L2 Distance",
                       "al1": "Adaptive L1", "al2": "Adaptive L2"}
        title = f"{metric_names.get(self.shape_metric.get(), self.shape_metric.get())} â€” {self.shape_param.get().upper()}"
        if self.shape_normalize.get():
            title += " (Normalized)"
        self.ax.set_title(title)
        
        self.fig.colorbar(im, ax=self.ax, label=('Correlation' if self.shape_metric.get() == 'xcorr' else 'Distance'))
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _export_matrix(self):
        if self.shape_data is None:
            messagebox.showinfo("No data", "No shape-comparison data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save matrix",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(f"# Shape Matrix\n")
                f.write(f"# Metric: {self.shape_metric.get()}\n")
                f.write(f"# Parameter: {self.shape_param.get().upper()}\n")
                f.write(f"# Normalized: {self.shape_normalize.get()}\n")
                f.write(f"# Files: {', '.join(self.shape_data['names'])}\n")
                f.write("# Format: file1 file2 value lag_bins\n#\n")
                
                matrix = self.shape_data['matrix']
                lag_matrix = self.shape_data['lag_matrix']
                names = self.shape_data['names']
                
                for i, ni in enumerate(names):
                    for j, nj in enumerate(names):
                        lag = int(lag_matrix[i, j]) if self.shape_metric.get() == "xcorr" else 0
                        f.write(f"{ni}\t{nj}\t{matrix[i, j]:.6f}\t{lag}\n")
            
            messagebox.showinfo("Export complete", f"Matrix saved to {filename}")
    
    def _export_csv(self):
        if self.shape_data is None:
            messagebox.showinfo("No data", "No shape-comparison data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save matrix as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            names = self.shape_data['names']
            df = pd.DataFrame(self.shape_data['matrix'], index=names, columns=names)
            df.to_csv(filename)
            
            messagebox.showinfo("Export complete", f"Matrix saved to {filename}")
    
    def get_text_output(self):
        if self.shape_data is None:
            return ""
        return format_shape_text(self.shape_data, self.shape_metric.get())
