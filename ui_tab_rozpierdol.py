import tkinter as tk
from tkinter import ttk
from analysis_rozpierdol import extract_overlay_data, format_overlay_text

class TabRozpierdol:
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
        
        self.mode_var = tk.StringVar(value='db')
        self.last_data = None
        
        self._build_ui()
    
    def _build_ui(self):
        ttk.Label(self.control_frame, text="Mode:").pack(side=tk.LEFT, padx=(0,5))
        for text, val in [("Mag (dB)", "db"), ("Mag (Lin)", "mag"), ("Phase (deg)", "phase")]:
            ttk.Radiobutton(self.control_frame, text=text, variable=self.mode_var,
                          value=val, command=self.update).pack(side=tk.LEFT, padx=2)
    
    def _get_file_params_map(self):
        result = {}
        for v, p, d in self.get_files():
            if not v.get():
                continue
            params = d.get('overlay_params', set())
            if params:
                result[p] = list(params)
        return result
    
    def update(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        
        file_params_map = self._get_file_params_map()
        
        if not file_params_map:
            ax.text(0.5, 0.5, "Right-click files and add S-parameters\nto show overlay plot",
                   ha="center", va="center", fontsize=12, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            self._update_legend_panel([])
            self.last_data = None
            return
        
        fmin, fmax, sstr = self.get_freq_range()
        mode = self.mode_var.get()
        data = extract_overlay_data(self.get_files(), sstr, file_params_map, mode)
        
        if not data:
            ax.text(0.5, 0.5, "No data available",
                   ha="center", va="center", fontsize=12, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            self._update_legend_panel([])
            self.last_data = None
            return
        
        self.last_data = data
        
        legend_items = []
        
        for entry in data:
            kwargs = {'label': entry['label']}
            if entry['color']:
                kwargs['color'] = entry['color']
            if entry['linewidth'] != 1.0:
                kwargs['linewidth'] = entry['linewidth']
            
            line = ax.plot(entry['freq'], entry['values'], **kwargs)[0]
            legend_items.append((entry['label'], line.get_color()))
        
        if mode == 'db':
            ylabel = 'Magnitude [dB]'
        elif mode == 'mag':
            ylabel = 'Magnitude [linear]'
        else:
            ylabel = 'Phase [deg]'
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_title('S-Parameter Overlay')
        ax.grid(True)
        
        if self.get_legend_on_plot():
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ncol = min(5, (len(labels) - 1) // 10 + 1)
                ax.legend(handles, labels, loc='best', ncol=ncol,
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
            
            color_label = tk.Label(row, text="\u25a0 ", foreground=color, font=("", 12))
            color_label.pack(side=tk.LEFT, padx=(0, 5))
            
            display_name = name if len(name) <= 30 else name[:27] + "..."
            ttk.Label(row, text=display_name, font=("", 9)).pack(side=tk.LEFT)
        
        self.legend_canvas.configure(scrollregion=self.legend_canvas.bbox("all"))
    
    def get_text_output(self):
        if self.last_data is None:
            return ""
        return format_overlay_text(self.last_data, self.mode_var.get())
