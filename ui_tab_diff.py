import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from analysis_diff import compute_diff, format_diff_text

class TabDiff:
    def __init__(self, parent, control_frame, fig, canvas,
                 legend_frame, legend_canvas,
                 get_files_func, get_freq_range_func,
                 get_legend_on_plot_func):
        self.parent = parent
        self.control_frame = control_frame
        self.fig = fig
        self.canvas = canvas
        self.legend_frame = legend_frame
        self.legend_canvas = legend_canvas
        self.get_files = get_files_func
        self.get_freq_range = get_freq_range_func
        self.get_legend_on_plot = get_legend_on_plot_func
        
        self.s11_var = tk.BooleanVar(value=True)
        self.s22_var = tk.BooleanVar(value=False)
        self.s12_var = tk.BooleanVar(value=False)
        self.s21_var = tk.BooleanVar(value=True)
        
        self.diff_pairs = []
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']
        
        self._build_ui()
    
    def _build_ui(self):
        ttk.Label(self.control_frame, text="Params:").pack(side=tk.LEFT, padx=(0,5))
        for name, var in [("S11", self.s11_var), ("S22", self.s22_var),
                          ("S12", self.s12_var), ("S21", self.s21_var)]:
            ttk.Checkbutton(self.control_frame, text=name, variable=var,
                           command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.control_frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Button(self.control_frame, text="Add diff", command=self._show_add_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.control_frame, text="Clear diffs", command=self._clear_diffs).pack(side=tk.LEFT, padx=2)
    
    def _get_selected_params(self):
        params = []
        if self.s11_var.get(): params.append('s11')
        if self.s22_var.get(): params.append('s22')
        if self.s12_var.get(): params.append('s12')
        if self.s21_var.get(): params.append('s21')
        return params
    
    def _get_available_files(self):
        fmin, fmax, sstr = self.get_freq_range()
        available = []
        
        for v, p, d in self.get_files():
            ext = Path(p).suffix.lower()
            if not (d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']):
                continue
            
            fname = d.get('custom_name') if d.get('is_average') else Path(p).stem
            available.append({'name': fname, 'path': p, 'data': d})
        
        return available, sstr
    
    def _load_network(self, file_info, sstr):
        from sparams_io import loadFile
        
        d = file_info['data']
        p = file_info['path']
        
        ntw_full = d.get('ntwk_full')
        if ntw_full is None:
            ntw_full = loadFile(p)
            d['ntwk_full'] = ntw_full
        
        if d.get('cached_range') != sstr:
            ntw = ntw_full[sstr]
            d['ntwk'] = ntw
            d['cached_range'] = sstr
        else:
            ntw = d['ntwk']
        
        return ntw
    
    def _show_add_dialog(self):
        available, sstr = self._get_available_files()
        
        if len(available) < 2:
            messagebox.showwarning("Not enough files", f"Need at least 2 S-param files loaded. Found: {len(available)}")
            return
        
        params = self._get_selected_params()
        if not params:
            messagebox.showwarning("No params", "Select at least one S-parameter")
            return
        
        dialog = tk.Toplevel(self.parent)
        dialog.title("Add Difference")
        dialog.transient(self.parent)
        dialog.grab_set()
        
        file_names = [f['name'] for f in available]
        
        ttk.Label(dialog, text="File A (minuend):", font=("", 10, "bold")).grid(row=0, column=0, padx=10, pady=(10,5), sticky="w")
        file_a_var = tk.StringVar(value=file_names[0] if file_names else "")
        file_a_combo = ttk.Combobox(dialog, textvariable=file_a_var, values=file_names, state="readonly", width=40)
        file_a_combo.grid(row=1, column=0, padx=10, pady=(0,10), sticky="ew")
        
        ttk.Label(dialog, text="File B (subtrahend):", font=("", 10, "bold")).grid(row=2, column=0, padx=10, pady=(10,5), sticky="w")
        file_b_var = tk.StringVar(value=file_names[1] if len(file_names) > 1 else "")
        file_b_combo = ttk.Combobox(dialog, textvariable=file_b_var, values=file_names, state="readonly", width=40)
        file_b_combo.grid(row=3, column=0, padx=10, pady=(0,10), sticky="ew")
        
        ttk.Label(dialog, text="Result: A - B", font=("", 9)).grid(row=4, column=0, padx=10, pady=5)
        
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=5, column=0, pady=15)
        
        def on_add():
            name_a = file_a_var.get()
            name_b = file_b_var.get()
            
            if not name_a or not name_b:
                messagebox.showwarning("Selection required", "Select both files")
                return
            
            if name_a == name_b:
                messagebox.showwarning("Same file", "Select two different files")
                return
            
            file_a = next((f for f in available if f['name'] == name_a), None)
            file_b = next((f for f in available if f['name'] == name_b), None)
            
            if not file_a or not file_b:
                messagebox.showerror("Error", "Could not find selected files")
                return
            
            try:
                ntw_a = self._load_network(file_a, sstr)
                ntw_b = self._load_network(file_b, sstr)
            except Exception as e:
                messagebox.showerror("Load error", f"Could not load networks: {e}")
                return
            
            diff_data = compute_diff(ntw_a, ntw_b, params)
            if diff_data is None:
                messagebox.showerror("Error", "Could not compute difference (frequency mismatch)")
                return
            
            self.diff_pairs.append({
                'name1': name_a,
                'name2': name_b,
                'diff_data': diff_data
            })
            
            dialog.destroy()
            self.update()
        
        ttk.Button(button_frame, text="Add", command=on_add).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        dialog.update_idletasks()
        x = dialog.winfo_screenwidth() // 2 - dialog.winfo_width() // 2
        y = dialog.winfo_screenheight() // 2 - dialog.winfo_height() // 2
        dialog.geometry(f"+{x}+{y}")
    
    def _clear_diffs(self):
        self.diff_pairs.clear()
        self.update()
    
    def update(self):
        self.fig.clf()
        
        params = self._get_selected_params()
        if not params:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Select at least one S-parameter",
                   ha="center", va="center", fontsize=12, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            self._update_legend_panel([])
            return
        
        if not self.diff_pairs:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No differences added\nClick 'Add diff' to select files",
                   ha="center", va="center", fontsize=12, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            self._update_legend_panel([])
            return
        
        n_params = len(params)
        axes = []
        for i, param in enumerate(params):
            if i == 0:
                ax = self.fig.add_subplot(n_params, 1, i+1)
            else:
                ax = self.fig.add_subplot(n_params, 1, i+1, sharex=axes[0])
            axes.append(ax)
        
        legend_items = []
        
        for ax, param in zip(axes, params):
            for i, pair in enumerate(self.diff_pairs):
                diff_data = pair['diff_data']
                if param not in diff_data:
                    continue
                
                freq = diff_data['freq']
                vals = diff_data[param]
                label = f"{pair['name1']} - {pair['name2']}"
                color = self.colors[i % len(self.colors)]
                
                ax.plot(freq, vals, label=label, color=color)
                
                if param == params[0]:
                    legend_items.append((label, color))
            
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_ylabel(f"D|{param.upper()}| [dB]")
            ax.set_title(f"{param.upper()} difference")
            ax.grid(True)
            
            if self.get_legend_on_plot() and param == params[-1]:
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ncol = min(3, (len(labels) - 1) // 5 + 1)
                    ax.legend(handles, labels, loc='best', ncol=ncol,
                             fontsize=8, framealpha=0.9)
        
        axes[-1].set_xlabel("Frequency [Hz]")
        self.fig.tight_layout()
        self.canvas.draw()
        
        self._update_legend_panel(legend_items)
    
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
            
            display_name = name if len(name) <= 30 else name[:27] + "..."
            ttk.Label(row, text=display_name, font=("", 9)).pack(side=tk.LEFT)
        
        self.legend_canvas.configure(scrollregion=self.legend_canvas.bbox("all"))
    
    def get_text_output(self):
        if not self.diff_pairs:
            return ""
        return format_diff_text(self.diff_pairs, self._get_selected_params())
