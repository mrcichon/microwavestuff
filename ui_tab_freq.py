import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt

from analysis_freq import extract_freq_data, find_extrema, format_freq_text

class TabFreq:
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
        
        self.s11_var = tk.BooleanVar(value=False)
        self.s22_var = tk.BooleanVar(value=False)
        self.s12_var = tk.BooleanVar(value=False)
        self.s21_var = tk.BooleanVar(value=False)
        
        self.extrema_enabled = tk.BooleanVar(value=False)
        self.extrema_minima = tk.BooleanVar(value=True)
        self.extrema_maxima = tk.BooleanVar(value=True)
        self.extrema_range_min = tk.DoubleVar(value=0.0)
        self.extrema_range_max = tk.DoubleVar(value=10.0)
        
        self.extrema_lines = []
        self.last_extrema = None
        
        self._build_ui()
        
    def _build_ui(self):
        ttk.Label(self.control_frame, text="Show:").pack(side=tk.LEFT, padx=(0,5))
        for name, var in [("S11", self.s11_var), ("S22", self.s22_var),
                          ("S12", self.s12_var), ("S21", self.s21_var)]:
            ttk.Checkbutton(self.control_frame, text=name, variable=var, 
                          command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.control_frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Checkbutton(self.control_frame, text="Show extrema", variable=self.extrema_enabled,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.control_frame, text="Min", variable=self.extrema_minima,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.control_frame, text="Max", variable=self.extrema_maxima,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(self.control_frame, text="Range [GHz]:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Entry(self.control_frame, textvariable=self.extrema_range_min, 
                 width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.control_frame, text="-").pack(side=tk.LEFT, padx=2)
        ttk.Entry(self.control_frame, textvariable=self.extrema_range_max,
                 width=5).pack(side=tk.LEFT, padx=2)
    
    def _get_selected_params(self):
        params = []
        if self.s11_var.get(): params.append('s11')
        if self.s22_var.get(): params.append('s22')
        if self.s12_var.get(): params.append('s12')
        if self.s21_var.get(): params.append('s21')
        return params
    
    def update(self):
        self.fig.clf()
        
        params = self._get_selected_params()
        if not params:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Select at least one parameter\nto show plot",
                   ha="center", va="center", fontsize=12, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            self._update_legend_panel([])
            return
        
        fmin, fmax, sstr = self.get_freq_range()
        use_db = self.get_scale_mode()
        files_data = extract_freq_data(self.get_files(), sstr, params, use_db)
        
        if not files_data:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No files selected",
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
        seen_names = set()
        
        for ax, param in zip(axes, params):
            for file_entry in files_data:
                if param not in file_entry['params']:
                    continue
                
                freq = file_entry['freq']
                values = file_entry['params'][param]
                name = file_entry['name']
                
                kwargs = {'label': name}
                if file_entry['color']:
                    kwargs['color'] = file_entry['color']
                if file_entry['linewidth'] != 1.0:
                    kwargs['linewidth'] = file_entry['linewidth']
                
                line = ax.plot(freq, values, **kwargs)[0]
                
                if name not in seen_names:
                    legend_items.append((name, line.get_color()))
                    seen_names.add(name)
            
            ylabel = f"|{param.upper()}| [dB]" if use_db else f"|{param.upper()}| [mag]"
            ax.set_ylabel(ylabel)
            ax.set_title(f"{param.upper()} magnitude")
            ax.grid(True)
            
            if self.get_legend_on_plot() and param == params[-1]:
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ncol = min(5, (len(labels) - 1) // 10 + 1)
                    ax.legend(handles, labels, loc='best', ncol=ncol, 
                            fontsize=8, framealpha=0.9)
        
        for line in self.extrema_lines:
            try:
                line.remove()
            except:
                pass
        self.extrema_lines.clear()
        
        if self.extrema_enabled.get():
            extrema_range = (self.extrema_range_min.get(), self.extrema_range_max.get())
            extrema = find_extrema(files_data, params, extrema_range,
                                  self.extrema_minima.get(), self.extrema_maxima.get())
            self.last_extrema = extrema
            
            for ax, param in zip(axes, params):
                param_extrema = [e for e in extrema if e['param'] == param]
                for ext in param_extrema:
                    color = 'red' if ext['type'] == 'max' else 'blue'
                    line = ax.axvline(x=ext['freq'], color=color, linestyle='--',
                                    alpha=0.3, linewidth=0.8)
                    self.extrema_lines.append(line)
        else:
            self.last_extrema = None
        
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
            
            display_name = name if len(name) <= 20 else name[:17] + "..."
            ttk.Label(row, text=display_name, font=("", 9)).pack(side=tk.LEFT)
        
        self.legend_canvas.configure(scrollregion=self.legend_canvas.bbox("all"))
    
    def get_text_output(self):
        if self.last_extrema is None or not self.extrema_enabled.get():
            return ""
        
        extrema_range = (self.extrema_range_min.get(), self.extrema_range_max.get())
        use_db = self.get_scale_mode()
        return format_freq_text(self.last_extrema, extrema_range, use_db)
