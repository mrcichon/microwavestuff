import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from itertools import cycle

from analysis_overlap import parse_frequency_file, find_overlaps, format_overlap_text

class TabOverlap:
    def __init__(self, parent, fig, ax, canvas,
                 get_files_func, get_freq_range_func, get_regex_tab_func):
        self.parent = parent
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.get_files = get_files_func
        self.get_freq_range = get_freq_range_func
        self.get_regex_tab = get_regex_tab_func
        
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
        regex_tab = self.get_regex_tab()
        
        if regex_tab is None or regex_tab.last_result is None:
            messagebox.showwarning("No regex data", 
                                 "Run regex analysis first in the Regex Highlighting tab.")
            return
        
        if not regex_tab.regex_ranges:
            messagebox.showwarning("No ranges found",
                                 "No monotonic ranges found in regex analysis.")
            return
        
        name = simpledialog.askstring("Range Name", 
                                     "Enter name for this range set:",
                                     initialvalue=f"regex_{regex_tab.regex_param.get()}")
        
        if not name:
            return
        
        ranges_ghz = [(start/1e9, end/1e9) for start, end in regex_tab.regex_ranges]
        self.overlap_data[name] = ranges_ghz
        self.update()
        messagebox.showinfo("Import successful", 
                          f"Imported {len(ranges_ghz)} range(s) as '{name}'")
    
    def _export_overlaps(self):
        if not self.last_result:
            messagebox.showinfo("No data", "No overlap data to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save overlap analysis",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(self.last_result)
            messagebox.showinfo("Export complete", f"Overlap analysis saved to {filename}")
    
    def _clear_plot(self):
        self.overlap_data = {}
        self.last_result = None
        self.update()
    
    def update(self):
        self.ax.clear()
        
        if not self.overlap_data:
            self.ax.text(0.5, 0.5, "Load range files or import from regex",
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.config(state=tk.DISABLED)
            return
        
        colors = cycle(['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'])
        y_pos = 0
        legend_handles = []
        all_freqs = []
        
        from itertools import combinations
        overlap_regions = []
        for (n1, r1), (n2, r2) in combinations(self.overlap_data.items(), 2):
            overlaps = find_overlaps(r1, r2)
            overlap_regions.extend(overlaps)
        
        for start, end in overlap_regions:
            self.ax.axvspan(start, end, color='green', alpha=0.2, zorder=0)
        
        for name, ranges in sorted(self.overlap_data.items()):
            color = next(colors)
            if not ranges:
                y_pos += 1
                continue
                
            for start, end in ranges:
                all_freqs.extend([start, end])
                rect = mpatches.Rectangle((start, y_pos), end - start, 0.8,
                                         facecolor=color, alpha=0.6, edgecolor='black', zorder=1)
                self.ax.add_patch(rect)
            
            legend_handles.append(mpatches.Patch(color=color, label=name, alpha=0.6))
            y_pos += 1
        
        self.ax.set_xlabel("Frequency [GHz]")
        self.ax.set_ylabel("Range Set")
        self.ax.set_title("Frequency Range Overlaps")
        self.ax.set_ylim(-0.5, len(self.overlap_data) + 0.5)
        self.ax.set_yticks(range(len(self.overlap_data)))
        self.ax.set_yticklabels(sorted(self.overlap_data.keys()))
        
        if all_freqs:
            freq_min = min(all_freqs)
            freq_max = max(all_freqs)
            margin = (freq_max - freq_min) * 0.05
            self.ax.set_xlim(freq_min - margin, freq_max + margin)
        
        self.ax.grid(True, axis='x')
        if legend_handles:
            self.ax.legend(handles=legend_handles, loc='best')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        result_text = format_overlap_text(self.overlap_data)
        self.last_result = result_text
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, result_text)
        self.info_text.config(state=tk.DISABLED)
    
    def get_text_output(self):
        return self.last_result if self.last_result else ""
