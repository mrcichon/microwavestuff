import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from itertools import cycle

from analysis_overlap import parse_frequency_file, find_overlaps, format_overlap_text

class TabOverlap:
    def __init__(self, parent, fig, ax, canvas,
                 get_files_func, get_freq_range_func):
        self.parent = parent
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.get_files = get_files_func
        self.get_freq_range = get_freq_range_func
        
        self.overlap_data = {}
        self.last_result = None
        
        self._build_ui()
        
    def _build_ui(self):
        frame = ttk.Frame(self.parent)
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame, text="Frequency Range Analysis:").pack(side=tk.LEFT, padx=(0,10))
        
        ttk.Button(frame, text="Load range files",
                  command=self._load_range_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Analyze from regex",
                  command=self._analyze_from_regex).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Export overlaps",
                  command=self._export_overlaps).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Clear plot",
                  command=self._clear_plot).pack(side=tk.LEFT, padx=2)
        
        # Info text goes below, takes remaining vertical space
        info_frame = ttk.LabelFrame(self.parent, text="Loaded Data")
        info_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=6, font=("Consolas", 8))
        self.info_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.info_text.config(state=tk.DISABLED)
        
        self.control_panel = frame


    def _load_range_files(self):
        files = filedialog.askopenfilenames(
            title="Select frequency range files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not files:
            return
        
        self.overlap_data = {}
        for filepath in files:
            key = Path(filepath).stem.split('_')[0]
            self.overlap_data[key] = parse_frequency_file(filepath)
        
        self.update()
    
    def _analyze_from_regex(self):
        # This would need access to regex tab data
        # For now, show a placeholder dialog
        messagebox.showinfo("Analyze from Regex",
                          "This would import ranges from the Regex tab.\nRun regex analysis first, then use this feature.")
    
    def _export_overlaps(self):
        if not self.overlap_data:
            messagebox.showinfo("No data", "No overlap data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save overlap analysis",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(format_overlap_text(self.overlap_data))
            
            messagebox.showinfo("Export complete", f"Overlap analysis saved to {filename}")
    
    def _clear_plot(self):
        self.overlap_data = {}
        self.update()
    
    def update(self):
        self.ax.clear()
        
        if not self.overlap_data:
            self.ax.text(0.5, 0.5, "Load frequency range files to visualize overlaps",
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, "No data loaded")
            self.info_text.config(state=tk.DISABLED)
            
            self.last_result = None
            return
        
        # Plot ranges
        colors = cycle(['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#17becf'])
        y_positions = {}
        legend_elements = []
        
        for i, (name, ranges) in enumerate(sorted(self.overlap_data.items())):
            color = next(colors)
            y_positions[name] = i
            for start, end in ranges:
                self.ax.plot([start, end], [i, i], color=color, linewidth=4)
            legend_elements.append(mpatches.Patch(color=color, label=f"{name}"))
        
        # Find and plot overlaps
        from itertools import combinations
        for (name1, r1), (name2, r2) in combinations(self.overlap_data.items(), 2):
            overlaps = find_overlaps(r1, r2)
            for s, e in overlaps:
                y1, y2 = y_positions[name1], y_positions[name2]
                ymin, ymax = sorted([y1, y2])
                self.ax.fill_betweenx([ymin - 0.15, ymax + 0.15], s, e, color='red', alpha=0.3)
        
        # Set labels and formatting
        self.ax.set_yticks(list(y_positions.values()))
        ylabels = list(y_positions.keys())
        if len(ylabels) > 15:
            self.ax.set_yticklabels(ylabels, fontsize=8)
        else:
            self.ax.set_yticklabels(ylabels)
        
        self.ax.set_xlabel("Frequency (GHz)")
        self.ax.set_title("Frequency ranges and overlaps")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        if len(legend_elements) <= 10:
            self.ax.legend(handles=legend_elements + [mpatches.Patch(color='red', alpha=0.3, label='Overlaps')])
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update info text
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, format_overlap_text(self.overlap_data))
        self.info_text.config(state=tk.DISABLED)
        
        self.last_result = self.overlap_data
    
    def get_text_output(self):
        if self.last_result is None:
            return ""
        return format_overlap_text(self.last_result)
