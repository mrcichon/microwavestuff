import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pathlib import Path

from analysis_integrate import compute_sparams, format_integration_text
from sparams_io import loadFile

class TabIntegrate:
    def __init__(self, parent, control_frame, fig, ax, canvas, get_files_func, get_freq_range_func, get_scale_mode_func):
        self.parent = parent
        self.control_frame = control_frame
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.get_files = get_files_func
        self.get_freq_range = get_freq_range_func
        self.get_scale_mode = get_scale_mode_func
        
        self.s11_var = tk.BooleanVar(value=True)
        self.s12_var = tk.BooleanVar(value=False)
        self.s21_var = tk.BooleanVar(value=True)
        self.s22_var = tk.BooleanVar(value=False)
        self.sort_var = tk.StringVar(value="name")
        self.asc_var = tk.BooleanVar(value=True)
        
        self.last_result = None
        
        self._build_ui()
        
    def _build_ui(self):
        
        ttk.Label(self.control_frame, text="Integrate:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Checkbutton(self.control_frame, text="S11", variable=self.s11_var, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.control_frame, text="S12", variable=self.s12_var, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.control_frame, text="S21", variable=self.s21_var, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.control_frame, text="S22", variable=self.s22_var, command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.control_frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(self.control_frame, text="Sort:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Radiobutton(self.control_frame, text="Name", value="name", variable=self.sort_var, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(self.control_frame, text="S11", value="s11", variable=self.sort_var, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(self.control_frame, text="S21", value="s21", variable=self.sort_var, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(self.control_frame, text="Total", value="total", variable=self.sort_var, command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.control_frame, text="↑", variable=self.asc_var, command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(self.control_frame, text="Export", command=self._export_txt).pack(side=tk.LEFT, padx=10)
        
            

    def _get_files_data(self):
        fmin, fmax, sstr = self.get_freq_range()
        
        files_data = []
        for v, p, d in self.get_files():
            if not v.get():
                continue
            
            ext = Path(p).suffix.lower()
            if not (d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']):
                continue
            
            try:
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
                for param in ['s11', 's12', 's21', 's22']:
                    try:
                        s_data = getattr(ntw, param)
                        freq = ntw.f
                        use_db = self.get_scale_mode()
                        values = s_data.s_db.flatten() if use_db else s_data.s_mag.flatten()
                        networks[param] = (freq, values)
                    except:
                        pass
                
                files_data.append({'name': fname, 'networks': networks})
                
            except Exception as e:
                continue
        
        return files_data
    
    def update(self):
        params = []
        if self.s11_var.get(): params.append('s11')
        if self.s12_var.get(): params.append('s12')
        if self.s21_var.get(): params.append('s21')
        if self.s22_var.get(): params.append('s22')
        
        if not params:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Select S-parameters to integrate", 
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            return
        
        files_data = self._get_files_data()
        
        if not files_data:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No valid data to integrate", 
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            return
        
        use_db = self.get_scale_mode()
        scale_type = 'db' if use_db else 'linear'
        result = compute_sparams(
            files_data,
            params,
            scale_type,
            self.sort_var.get(),
            self.asc_var.get()
        )
        
        self.last_result = result
        self._plot_result(result)
    
    def _plot_result(self, result):
        self.ax.clear()
        
        sorted_names = list(result['results'].keys())
        n_files = len(sorted_names)
        n_params = len(result['params'])
        x = np.arange(n_files)
        width = 0.8 / n_params
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, param in enumerate(result['params']):
            values = [result['results'][fname].get(param, 0) for fname in sorted_names]
            self.ax.bar(x + i * width, values, width, label=param.upper(), color=colors[i])
        
        self.ax.set_xlabel('Files')
        self.ax.set_ylabel('Integrated value')
        
        sort_label = "name" if result['sort_by'] == "name" else f"{result['sort_by'].upper()} {'Ã¢â€ â€˜' if self.asc_var.get() else 'Ã¢â€ â€œ'}"
        self.ax.set_title(f"Integration of S-parameters ({result['scale_type']} scale, sorted by {sort_label})")
        
        self.ax.set_xticks(x + width * (n_params - 1) / 2)
        self.ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def get_text_output(self):
        if self.last_result is None:
            return ""
        fmin, fmax, _ = self.get_freq_range()
        return format_integration_text(self.last_result, fmin, fmax)
    
    def _export_txt(self):
        if self.last_result is None:
            messagebox.showinfo("No data", "No integration data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save integration results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filename:
            return
        
        with open(filename, 'w') as f:
            f.write("# S-parameter Integration Results\n")
            f.write(f"# Scale: {self.last_result['scale_type']}\n")
            fmin, fmax, _ = self.get_freq_range()
            f.write(f"# Frequency range: {fmin}-{fmax} GHz\n")
            f.write("# Format: filename parameter integral_value\n")
            f.write("#\n")
            
            for fname, params in self.last_result['results'].items():
                for param, value in params.items():
                    f.write(f"{fname}\t{param}\t{value:.6e}\n")
        
        messagebox.showinfo("Export complete", f"Results saved to {filename}")
    
    def _export_csv(self):
        if self.last_result is None:
            messagebox.showinfo("No data", "No integration data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save integration results as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filename:
            return
        
        import pandas as pd
        df = pd.DataFrame.from_dict(self.last_result['results'], orient='index')
        df.to_csv(filename)
        
        messagebox.showinfo("Export complete", f"Results saved to {filename}")
