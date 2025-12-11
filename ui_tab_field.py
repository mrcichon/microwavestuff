import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from analysis_field import load_field_csv, compute_layer_stats, compute_global_stats

class TabField:
    def __init__(self, parent, control_frame, fig, ax, canvas):
        self.parent = parent
        self.control_frame = control_frame
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        
        self.layers = None
        self.global_stats = None
        self.current_layer = 0
        self.text_annotations = []
        
        self.cmap = LinearSegmentedColormap.from_list('field', ['red', 'yellow', 'green'])
        
        self._build_ui()
    
    def _build_ui(self):
        ttk.Button(self.control_frame, text="Load CSV", 
                   command=self._load_csv).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.control_frame, text="Layer:").pack(side=tk.LEFT, padx=(20, 5))
        
        ttk.Button(self.control_frame, text="<", width=3,
                   command=self._prev_layer).pack(side=tk.LEFT)
        
        self.layer_var = tk.IntVar(value=0)
        self.layer_slider = ttk.Scale(self.control_frame, from_=0, to=0,
                                       variable=self.layer_var, orient=tk.HORIZONTAL,
                                       length=200, command=self._on_slider)
        self.layer_slider.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.control_frame, text=">", width=3,
                   command=self._next_layer).pack(side=tk.LEFT)
        
        self.layer_label = ttk.Label(self.control_frame, text="(0 / 0)")
        self.layer_label.pack(side=tk.LEFT, padx=10)
        
        stats_frame = ttk.LabelFrame(self.parent, text="Statistics")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=4, font=("Consolas", 9))
        self.stats_text.pack(fill=tk.X, padx=5, pady=5)
        self.stats_text.config(state=tk.DISABLED)
    
    def _load_csv(self):
        filepath = filedialog.askopenfilename(
            title="Select field data CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        rows = simpledialog.askinteger("Grid dimensions", "Rows per layer:", 
                                        initialvalue=10, minvalue=1)
        if not rows:
            return
        
        cols = simpledialog.askinteger("Grid dimensions", "Columns per layer:",
                                        initialvalue=11, minvalue=1)
        if not cols:
            return
        
        try:
            self.layers = load_field_csv(filepath, rows, cols)
            self.global_stats = compute_global_stats(self.layers)
            self.current_layer = 0
            
            n_layers = self.layers.shape[0]
            self.layer_slider.configure(to=n_layers - 1)
            self.layer_var.set(0)
            
            self.update()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")
    
    def _prev_layer(self):
        if self.layers is None:
            return
        self.current_layer = max(0, self.current_layer - 1)
        self.layer_var.set(self.current_layer)
        self.update()
    
    def _next_layer(self):
        if self.layers is None:
            return
        self.current_layer = min(self.layers.shape[0] - 1, self.current_layer + 1)
        self.layer_var.set(self.current_layer)
        self.update()
    
    def _on_slider(self, val):
        if self.layers is None:
            return
        self.current_layer = int(float(val))
        self.update()
    
    def update(self):
        self.ax.clear()
        for ann in self.text_annotations:
            ann.remove()
        self.text_annotations = []
        
        if self.layers is None:
            self.ax.text(0.5, 0.5, "Load a CSV file to display field data",
                        ha='center', va='center', fontsize=12, color='gray',
                        transform=self.ax.transAxes)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            return
        
        layer = self.layers[self.current_layer]
        n_layers = self.layers.shape[0]
        rows, cols = layer.shape
        
        abs_max = self.global_stats['abs_max']
        if abs_max == 0:
            abs_max = 1
        normalized = np.abs(layer) / abs_max
        
        im = self.ax.imshow(normalized, cmap=self.cmap, vmin=0, vmax=1,
                           aspect='auto', origin='upper')
        
        for i in range(rows):
            for j in range(cols):
                val = layer[i, j]
                brightness = normalized[i, j]
                text_color = 'white' if brightness < 0.5 else 'black'
                ann = self.ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                                  fontsize=7, color=text_color, fontweight='bold')
                self.text_annotations.append(ann)
        
        self.ax.set_xticks(range(cols))
        self.ax.set_yticks(range(rows))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(f'Layer {self.current_layer}')
        
        self.layer_label.configure(text=f"({self.current_layer} / {n_layers - 1})")
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        self._update_stats()
    
    def _update_stats(self):
        if self.layers is None:
            return
        
        layer = self.layers[self.current_layer]
        ls = compute_layer_stats(layer)
        gs = self.global_stats
        
        text = (
            f"Layer {self.current_layer}:  "
            f"Mean: {ls['mean']:.4f}   Variance: {ls['variance']:.4f}   "
            f"Max Δ: {ls['max_diff']:.4f}   Range: [{ls['min']:.4f}, {ls['max']:.4f}]\n"
            f"Global (all layers):  "
            f"Min: {gs['min']:.4f}   Max: {gs['max']:.4f}   "
            f"Range: {gs['range']:.4f}   |Max|: {gs['abs_max']:.4f}"
        )
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, text)
        self.stats_text.config(state=tk.DISABLED)
