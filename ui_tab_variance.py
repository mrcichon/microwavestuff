import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pathlib import Path

from analysis_variance import compute_variance, format_variance_text

class TabVariance:
    def __init__(self, parent, fig, ax, canvas,
                 get_files_func, get_freq_range_func):
        self.parent = parent
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.get_files = get_files_func
        self.get_freq_range = get_freq_range_func
        
        self.var_s11 = tk.BooleanVar(value=True)
        self.var_s12 = tk.BooleanVar(value=False)
        self.var_s21 = tk.BooleanVar(value=True)
        self.var_s22 = tk.BooleanVar(value=False)
        self.var_mag = tk.BooleanVar(value=True)
        self.var_phase = tk.BooleanVar(value=True)
        self.var_detrend = tk.BooleanVar(value=True)
        
        self.var_data = None
        self.var_markers = []
        self.var_marker_text = []
        
        self._build_ui()
        
    def _build_ui(self):
        frame = ttk.Frame(self.parent)
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame, text="Params:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Checkbutton(frame, text="S11", variable=self.var_s11,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(frame, text="S12", variable=self.var_s12,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(frame, text="S21", variable=self.var_s21,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(frame, text="S22", variable=self.var_s22,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Checkbutton(frame, text="Magnitude", variable=self.var_mag,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(frame, text="Phase", variable=self.var_phase,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Checkbutton(frame, text="Detrend phase",
                       variable=self.var_detrend, command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(frame, text="Clear markers",
                  command=self._clear_markers).pack(side=tk.LEFT, padx=10)
        ttk.Button(frame, text="Export",
                  command=self._export_variance).pack(side=tk.LEFT, padx=2)
        
        self.control_panel = frame


    def _get_files_data(self):
        fmin, fmax, sstr = self.get_freq_range()
        
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
                    
                    networks = {}
                    for sparam in ['s11', 's12', 's21', 's22']:
                        if hasattr(ntw, sparam):
                            s_data = getattr(ntw, sparam)
                            networks[sparam] = (ntw.f, s_data.s.flatten())
                    
                    files_data.append({
                        'name': fname,
                        'freq': ntw.f,
                        'networks': networks
                    })
                    
                except Exception:
                    pass
        
        return files_data
    
    def update(self):
        self.ax.clear()
        
        # Build parameter list
        param_list = []
        component_types = []
        
        if self.var_s11.get():
            if self.var_mag.get():
                param_list.append('s11_mag')
                component_types.append('mag')
            if self.var_phase.get():
                param_list.append('s11_phase')
                component_types.append('phase')
        
        if self.var_s12.get():
            if self.var_mag.get():
                param_list.append('s12_mag')
                component_types.append('mag')
            if self.var_phase.get():
                param_list.append('s12_phase')
                component_types.append('phase')
        
        if self.var_s21.get():
            if self.var_mag.get():
                param_list.append('s21_mag')
                component_types.append('mag')
            if self.var_phase.get():
                param_list.append('s21_phase')
                component_types.append('phase')
        
        if self.var_s22.get():
            if self.var_mag.get():
                param_list.append('s22_mag')
                component_types.append('mag')
            if self.var_phase.get():
                param_list.append('s22_phase')
                component_types.append('phase')
        
        if not param_list:
            self.ax.text(0.5, 0.5, "Select S-parameters and at least one component type",
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            self.var_data = None
            return
        
        files_data = self._get_files_data()
        
        if len(files_data) < 2:
            self.ax.text(0.5, 0.5, "Need at least 2 files selected",
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            self.var_data = None
            return
        
        # Compute variance
        self.var_data = compute_variance(files_data, param_list, component_types,
                                        self.var_detrend.get())
        
        if self.var_data is None:
            self.ax.text(0.5, 0.5, "Error computing variance",
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            return
        
        # Plot stacked area chart
        freqs = self.var_data['frequencies']
        var_contrib = self.var_data['variance_contribution']
        n_params = self.var_data['n_params']
        
        # Create labels
        param_labels = []
        for param in self.var_data['param_list']:
            parts = param.split('_')
            s_param = parts[0].upper()
            comp_type = parts[1].capitalize()
            param_labels.append(f'{s_param} {comp_type}')
        
        # Prepare data for stackplot
        y = np.zeros((n_params, len(freqs)))
        for i in range(n_params):
            y[i] = var_contrib[:, i]
        
        # Use distinct colors
        if n_params <= 8:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'][:n_params]
        else:
            import matplotlib.cm as cm
            colors = cm.tab20(np.linspace(0, 1, n_params))
        
        self.ax.stackplot(freqs, y, labels=param_labels[:n_params],
                         colors=colors, alpha=0.8)
        
        # Mark high variance regions
        mean_var = np.mean(self.var_data['total_variance'])
        std_var = np.std(self.var_data['total_variance'])
        high_var_mask = self.var_data['total_variance'] > mean_var + 2 * std_var
        
        if np.any(high_var_mask):
            high_var_freqs = freqs[high_var_mask]
            for f in high_var_freqs:
                self.ax.axvline(x=f, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        self.ax.set_xlabel('Frequency [Hz]')
        self.ax.set_ylabel('Variance Contribution (%)')
        self.ax.set_title(f'Normalized Variance Contribution by Component ({self.var_data["file_count"]} files)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(0, 105)
        
        if n_params <= 8:
            self.ax.legend(loc='upper right', fontsize=9)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _clear_markers(self):
        for m in self.var_markers:
            try:
                m['marker'].remove()
                m['text'].remove()
            except:
                pass
        self.var_markers.clear()
        self.var_marker_text.clear()
        self.canvas.draw()
    
    def _export_variance(self):
        if self.var_data is None:
            messagebox.showinfo("No data", "No variance data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save variance data",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write("# Variance Analysis (Normalized Data)\n")
                f.write(f"# Normalization: {self.var_data['normalization_info']}\n")
                f.write(f"# Parameters: {', '.join(self.var_data['param_list'])}\n")
                f.write(f"# Number of files: {self.var_data['file_count']}\n")
                f.write(f"# Mean total variance: {np.mean(self.var_data['total_variance']):.6f}\n")
                f.write(f"# Std total variance: {np.std(self.var_data['total_variance']):.6f}\n")
                f.write("# Frequency[Hz] Total_Variance Component_Variances...\n")
                
                for i, freq in enumerate(self.var_data['frequencies']):
                    f.write(f"{freq} {self.var_data['total_variance'][i]}")
                    for j in range(self.var_data['n_params']):
                        f.write(f" {self.var_data['variance_by_param'][i, j]}")
                    f.write("\n")
            
            messagebox.showinfo("Export complete", f"Variance data saved to {filename}")
    
    def get_text_output(self):
        if self.var_data is None:
            return ""
        return format_variance_text(self.var_data)
