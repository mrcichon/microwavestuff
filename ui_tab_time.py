# ui_tab_time.py

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from analysis_time import extract_time_data, apply_time_gate, find_time_extrema, find_gated_extrema, format_time_text

class TabTime:
    def __init__(self, parent, fig, canvas,
                 legend_frame, legend_canvas,
                 get_files_func, get_freq_range_func,
                 get_legend_on_plot_func):
        self.parent = parent
        self.fig = fig
        self.canvas = canvas
        self.legend_frame = legend_frame
        self.legend_canvas = legend_canvas
        
        self.get_files = get_files_func
        self.get_freq_range = get_freq_range_func
        self.get_legend_on_plot = get_legend_on_plot_func
        
        self.td_param_var = tk.StringVar(value="s11")
        self.gate_chk_var = tk.BooleanVar(value=False)
        self.gate_center_var = tk.DoubleVar(value=5.0)
        self.gate_span_var = tk.DoubleVar(value=0.5)
        
        self.extrema_enabled = tk.BooleanVar(value=False)
        self.extrema_minima = tk.BooleanVar(value=True)
        self.extrema_maxima = tk.BooleanVar(value=True)
        self.extrema_range_min = tk.DoubleVar(value=0.0)
        self.extrema_range_max = tk.DoubleVar(value=10.0)
        
        self.extrema_lines = []
        self.last_result = None
        
        self._build_ui()
        
    def _build_ui(self):
        frame = ttk.Frame(self.parent)
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame, text="TDG:").pack(side=tk.LEFT, padx=(0,5))
        for n in ("s11", "s12", "s21", "s22"):
            ttk.Radiobutton(frame, text=n.upper(), value=n, 
                           variable=self.td_param_var, command=self.update).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Checkbutton(frame, text="Gating", variable=self.gate_chk_var, 
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame, text="Center[ns]").pack(side=tk.LEFT, padx=(5,2))
        ttk.Entry(frame, textvariable=self.gate_center_var, 
                 width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame, text="Span[ns]").pack(side=tk.LEFT, padx=(5,2))
        ttk.Entry(frame, textvariable=self.gate_span_var, 
                 width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Checkbutton(frame, text="Extrema", 
                       variable=self.extrema_enabled,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(frame, text="Min", variable=self.extrema_minima,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(frame, text="Max", variable=self.extrema_maxima,
                       command=self.update).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frame, textvariable=self.extrema_range_min, 
                 width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame, text="-").pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=self.extrema_range_max,
                 width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(frame, text="GHz").pack(side=tk.LEFT, padx=(0,5))
        
        self.control_panel = frame


    def update(self):
        self.fig.clear()
        
        fmin, fmax, sstr = self.get_freq_range()
        td_param = self.td_param_var.get()
        
        raw_data = extract_time_data(self.get_files(), sstr, td_param)
        
        if not raw_data:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No valid files selected",
                   ha="center", va="center", fontsize=12, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            self._update_legend_panel([])
            return
        
        gated_data = None
        if self.gate_chk_var.get():
            gated_data = apply_time_gate(raw_data, td_param,
                                        self.gate_center_var.get(),
                                        self.gate_span_var.get())
        
        extrema = []
        if self.extrema_enabled.get():
            extrema_range = (self.extrema_range_min.get(), self.extrema_range_max.get())
            extrema.extend(find_time_extrema(
                raw_data, td_param, extrema_range,
                self.extrema_minima.get(), self.extrema_maxima.get()
            ))
            
            if gated_data:
                extrema.extend(find_gated_extrema(
                    gated_data, extrema_range, td_param,
                    self.extrema_minima.get(), self.extrema_maxima.get()
                ))
        
        self.last_result = {
            'raw': raw_data,
            'gated': gated_data,
            'extrema': extrema,
            'param': td_param
        }
        
        self._plot_result()
    
    def _plot_result(self):
        axes = self.fig.subplots(3, 1, sharex=False)
        axF, axTR, axTG = axes
        
        prm = self.last_result['param'].upper()
        legend_items = []
        seen_labels = set()
        
        for data in self.last_result['raw']:
            kwargs = {'label': data['name']}
            if data['color']:
                kwargs['color'] = data['color']
            if data['linewidth'] != 1.0:
                kwargs['linewidth'] = data['linewidth']
            
            line = axF.plot(data['freq'], data['freq_data'], **kwargs)[0]
            axTR.plot(data['t_ns'], data['time_data'], **kwargs)
            
            if data['name'] not in seen_labels:
                legend_items.append((data['name'], line.get_color()))
                seen_labels.add(data['name'])
        
        axF.set_ylabel(f"|{prm}| [dB]")
        axF.set_title(f"{prm} (frequency domain - raw)")
        axF.grid(True)
        
        axTR.set_ylabel(f"{prm} TD [dB]")
        axTR.set_xlabel("Time [ns]")
        axTR.set_title(f"{prm} (time domain - raw)")
        axTR.grid(True)
        axTR.set_xlim(0, 50)
        
        if self.last_result['gated']:
            for data in self.last_result['gated']:
                kwargs = {'label': data['name']}
                if data['color']:
                    kwargs['color'] = data['color']
                if data['linewidth'] != 1.0:
                    kwargs['linewidth'] = data['linewidth']
                axTG.plot(data['freq'], data['gated_data'], **kwargs)
        
        axTG.set_ylabel(f"{prm} [dB]")
        axTG.set_xlabel("Frequency [Hz]")
        axTG.set_title(f"{prm} (frequency domain - gated)")
        axTG.grid(True)
        
        for line in self.extrema_lines:
            try:
                line.remove()
            except:
                pass
        self.extrema_lines.clear()
        
        if self.last_result['extrema']:
            for ext in self.last_result['extrema']:
                ax = axTG if ext['domain'] == 'freq_gated' else axF
                color = 'red' if ext['type'] == 'max' else 'blue'
                line = ax.axvline(x=ext['freq'], color=color, linestyle='--',
                                alpha=0.3, linewidth=0.8)
                self.extrema_lines.append(line)
        
        if self.get_legend_on_plot():
            handles, labels = axF.get_legend_handles_labels()
            if labels:
                ncol = min(5, (len(labels) - 1) // 10 + 1)
                axF.legend(handles, labels, loc='best', ncol=ncol,
                          fontsize=8, framealpha=0.9)
        
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
            
            color_label = tk.Label(row, text="■ ", foreground=color, font=("", 12))
            color_label.pack(side=tk.LEFT, padx=(0, 5))
            
            display_name = name if len(name) <= 20 else name[:17] + "..."
            ttk.Label(row, text=display_name, font=("", 9)).pack(side=tk.LEFT)
        
        self.legend_canvas.configure(scrollregion=self.legend_canvas.bbox("all"))
    
    def get_text_output(self):
        if self.last_result is None:
            return ""
        
        lines = []
        prm = self.last_result['param']
        lines.append(f"Time Domain Analysis: {prm.upper()}")
        
        if self.gate_chk_var.get():
            center = self.gate_center_var.get()
            span = self.gate_span_var.get()
            lines.append(f"Gating: ON (Center: {center:.2f} ns, Span: {span:.2f} ns)")
        else:
            lines.append("Gating: OFF")
        
        if self.last_result['extrema']:
            extrema_range = (self.extrema_range_min.get(), self.extrema_range_max.get())
            extrema_text = format_time_text(self.last_result['extrema'], extrema_range)
            if extrema_text:
                lines.append("")
                lines.append(extrema_text)
        
        return "\n".join(lines)
