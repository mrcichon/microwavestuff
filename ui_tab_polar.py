import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from analysis_polar import extract_theta_phi_series, prepare_polar_series
from sparams_io import parse_polar_rms, parse_polar_pustelnik, parse_theta_phi_file

class TabPolar:
    def __init__(self, parent, fig, ax, canvas):
        self.parent = parent
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        
        self.mode = tk.StringVar(value="rms")
        self.scale_min = tk.DoubleVar(value=-10.0)
        self.scale_max = tk.DoubleVar(value=10.0)
        self.scale_step = tk.DoubleVar(value=2.5)
        
        self.rms_data = None
        self.pustelnik_data = None
        self.theta_phi_files = []
        self.combined_items = []
        
        self.rms_checkboxes = {}
        
        self._build_ui()
    
    def _build_ui(self):
        control_bar = ttk.Frame(self.parent)
        control_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_bar, text="Mode:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Radiobutton(control_bar, text="RMS/Circular", value="rms",
                       variable=self.mode, command=self._switch_mode).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(control_bar, text="Theta/Phi", value="theta_phi",
                       variable=self.mode, command=self._switch_mode).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(control_bar, text="Combined", value="combined",
                       variable=self.mode, command=self._switch_mode).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(control_bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(control_bar, text="Scale [dB]:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Label(control_bar, text="Min:").pack(side=tk.LEFT)
        ttk.Entry(control_bar, textvariable=self.scale_min, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(control_bar, text="Max:").pack(side=tk.LEFT)
        ttk.Entry(control_bar, textvariable=self.scale_max, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(control_bar, text="Step:").pack(side=tk.LEFT)
        ttk.Entry(control_bar, textvariable=self.scale_step, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_bar, text="Update", command=self.update).pack(side=tk.LEFT, padx=10)
        
        second_bar = ttk.Frame(self.parent)
        second_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0,5))
        
        self.mode_bars = {}
        self.mode_bars["rms"] = self._build_rms_bar(second_bar)
        self.mode_bars["theta_phi"] = self._build_tp_bar(second_bar)
        self.mode_bars["combined"] = self._build_combined_bar(second_bar)
        
        list_frame = ttk.Frame(self.parent)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        
        ttk.Label(list_frame, text="Files:").pack(anchor="w")
        
        scroll = ttk.Scrollbar(list_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas = tk.Canvas(list_frame, yscrollcommand=scroll.set, width=500, height=400)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=canvas.yview)
        
        self.file_frame = ttk.Frame(canvas)
        canvas.create_window((0,0), window=self.file_frame, anchor="nw")
        
        self.file_canvas = canvas
        
        def update_scroll(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.file_frame.bind("<Configure>", update_scroll)
        
        self._switch_mode()
    
    def _build_rms_bar(self, parent):
        frame = ttk.Frame(parent)
        
        ttk.Button(frame, text="Load RMS", command=self._load_rms).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Load Pustelnik", command=self._load_pustelnik).pack(side=tk.LEFT, padx=2)
        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.rms_checkbox_container = ttk.Frame(frame)
        self.rms_checkbox_container.pack(side=tk.LEFT, fill=tk.X)
        
        return frame
    
    def _build_tp_bar(self, parent):
        frame = ttk.Frame(parent)
        
        ttk.Button(frame, text="Add files", command=self._add_theta_phi).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Remove selected", command=self._remove_selected_tp).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Clear", command=self._clear_tp).pack(side=tk.LEFT, padx=2)
        
        return frame
    
    def _build_combined_bar(self, parent):
        frame = ttk.Frame(parent)
        
        ttk.Button(frame, text="Add RMS", command=self._add_combined_rms).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Add Theta/Phi", command=self._add_combined_tp).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Add Pustelnik", command=self._add_combined_pust).pack(side=tk.LEFT, padx=2)
        ttk.Separator(frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(frame, text="Remove selected", command=self._remove_selected_combined).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Clear", command=self._clear_combined).pack(side=tk.LEFT, padx=2)
        
        return frame
    
    def _switch_mode(self):
        for bar in self.mode_bars.values():
            bar.pack_forget()
        
        mode = self.mode.get()
        self.mode_bars[mode].pack(side=tk.LEFT, fill=tk.X)
        
        if mode == "rms":
            self.scale_min.set(-10.0)
            self.scale_max.set(10.0)
            self.scale_step.set(2.5)
        elif mode == "theta_phi":
            self.scale_min.set(-40.0)
            self.scale_max.set(20.0)
            self.scale_step.set(5.0)
        else:
            self.scale_min.set(-20.0)
            self.scale_max.set(15.0)
            self.scale_step.set(5.0)
        
        self._update_file_list()
        self.update()
    
    def _load_rms(self):
        path = filedialog.askopenfilename(
            title="Select RMS file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            try:
                self.rms_data = parse_polar_rms(path)
                self._rebuild_rms_checkboxes()
                self._update_file_list()
                self.update()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load RMS:\n{str(e)[:100]}")
    
    def _load_pustelnik(self):
        path = filedialog.askopenfilename(
            title="Select Pustelnik file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            try:
                self.pustelnik_data = parse_polar_pustelnik(path)
                self._rebuild_rms_checkboxes()
                self._update_file_list()
                self.update()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Pustelnik:\n{str(e)[:100]}")
    
    def _rebuild_rms_checkboxes(self):
        for widget in self.rms_checkbox_container.winfo_children():
            widget.destroy()
        
        self.rms_checkboxes = {}
        
        if self.rms_data:
            pol_type = self.rms_data.get("polarization", "linear")
            
            if pol_type == "circular":
                for plane in ["YZ", "ZX", "XY"]:
                    for pol in ["RHCP", "LHCP", "RL"]:
                        if self.rms_data[plane].get(pol) is not None:
                            var = tk.BooleanVar(value=pol != "RL")
                            key = f"{plane}_{pol}"
                            self.rms_checkboxes[key] = var
                            ttk.Checkbutton(self.rms_checkbox_container, text=f"{plane}_{pol}",
                                          variable=var, command=self.update).pack(side=tk.LEFT, padx=2)
            else:
                for plane in ["YZ", "ZX", "XY"]:
                    if self.rms_data[plane].get("HV") is not None:
                        var = tk.BooleanVar(value=True)
                        key = f"{plane}_HV"
                        self.rms_checkboxes[key] = var
                        ttk.Checkbutton(self.rms_checkbox_container, text=f"{plane}_HV",
                                      variable=var, command=self.update).pack(side=tk.LEFT, padx=2)
        
        if self.pustelnik_data:
            var = tk.BooleanVar(value=True)
            self.rms_checkboxes["pustelnik"] = var
            ttk.Checkbutton(self.rms_checkbox_container, text="Pustelnik",
                          variable=var, command=self.update).pack(side=tk.LEFT, padx=2)
    
    def _add_theta_phi(self):
        paths = filedialog.askopenfilenames(
            title="Select Theta/Phi files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        for path in paths:
            try:
                df = parse_theta_phi_file(path)
                theta, db, used = extract_theta_phi_series(df)
                theta = (360.0 - theta) % 360.0
                
                var = tk.BooleanVar(value=True)
                self.theta_phi_files.append({
                    "name": Path(path).stem,
                    "theta": theta,
                    "db": db,
                    "used": used,
                    "var": var
                })
            except Exception as e:
                messagebox.showerror("Error", f"{Path(path).name}:\n{str(e)[:100]}")
        
        self._update_file_list()
        self.update()
    
    def _remove_tp_index(self, idx):
        if 0 <= idx < len(self.theta_phi_files):
            self.theta_phi_files.pop(idx)
            self._update_file_list()
            self.update()
    
    def _remove_selected_tp(self):
        self.theta_phi_files = [f for f in self.theta_phi_files if f['var'].get()]
        self._update_file_list()
        self.update()
    
    def _clear_tp(self):
        self.theta_phi_files = []
        self._update_file_list()
        self.update()
    
    def _add_combined_rms(self):
        path = filedialog.askopenfilename(
            title="Select RMS file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            try:
                data = parse_polar_rms(path)
                name = Path(path).stem
                pol_type = data.get("polarization", "linear")
                
                if pol_type == "circular":
                    for plane in ["YZ", "ZX", "XY"]:
                        for pol in ["RHCP", "LHCP"]:
                            if data[plane].get(pol) is not None:
                                var = tk.BooleanVar(value=True)
                                self.combined_items.append({
                                    "name": f"{name}_{plane}_{pol}",
                                    "angle": data["angle_deg"],
                                    "values": data[plane][pol],
                                    "var": var
                                })
                else:
                    for plane in ["YZ", "ZX", "XY"]:
                        if data[plane].get("HV") is not None:
                            var = tk.BooleanVar(value=True)
                            self.combined_items.append({
                                "name": f"{name}_{plane}_HV",
                                "angle": data["angle_deg"],
                                "values": data[plane]["HV"],
                                "var": var
                            })
                
                self._update_file_list()
                self.update()
            except Exception as e:
                messagebox.showerror("Error", str(e)[:100])
    
    def _add_combined_tp(self):
        paths = filedialog.askopenfilenames(
            title="Select Theta/Phi files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        for path in paths:
            try:
                df = parse_theta_phi_file(path)
                theta, db, used = extract_theta_phi_series(df)
                theta = (360.0 - theta) % 360.0
                
                var = tk.BooleanVar(value=True)
                self.combined_items.append({
                    "name": Path(path).stem,
                    "angle": theta,
                    "values": db,
                    "var": var
                })
            except Exception as e:
                messagebox.showerror("Error", str(e)[:100])
        
        self._update_file_list()
        self.update()
    
    def _add_combined_pust(self):
        path = filedialog.askopenfilename(
            title="Select Pustelnik file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            try:
                data = parse_polar_pustelnik(path)
                var = tk.BooleanVar(value=True)
                self.combined_items.append({
                    "name": Path(path).stem,
                    "angle": data["angle_deg"],
                    "values": data["HV"],
                    "var": var
                })
                self._update_file_list()
                self.update()
            except Exception as e:
                messagebox.showerror("Error", str(e)[:100])
    
    def _remove_combined_index(self, idx):
        if 0 <= idx < len(self.combined_items):
            self.combined_items.pop(idx)
            self._update_file_list()
            self.update()
    
    def _remove_selected_combined(self):
        self.combined_items = [item for item in self.combined_items if item['var'].get()]
        self._update_file_list()
        self.update()
    
    def _clear_combined(self):
        self.combined_items = []
        self._update_file_list()
        self.update()
    
    def _update_file_list(self):
        for widget in self.file_frame.winfo_children():
            widget.destroy()
        
        mode = self.mode.get()
        
        if mode == "theta_phi":
            for i, item in enumerate(self.theta_phi_files):
                row = ttk.Frame(self.file_frame)
                row.pack(fill=tk.X, pady=1)
                
                ttk.Checkbutton(row, text=f"{item['name']} ({item['used']})",
                              variable=item['var'], command=self.update).pack(side=tk.LEFT)
                ttk.Button(row, text="Remove", width=8,
                          command=lambda idx=i: self._remove_tp_index(idx)).pack(side=tk.RIGHT)
        
        elif mode == "combined":
            for i, item in enumerate(self.combined_items):
                row = ttk.Frame(self.file_frame)
                row.pack(fill=tk.X, pady=1)
                
                ttk.Checkbutton(row, text=item['name'],
                              variable=item['var'], command=self.update).pack(side=tk.LEFT)
                ttk.Button(row, text="Remove", width=8,
                          command=lambda idx=i: self._remove_combined_index(idx)).pack(side=tk.RIGHT)
        
        self.file_canvas.configure(scrollregion=self.file_canvas.bbox("all"))
    
    def update(self):
        self.ax.clear()
        
        min_db = self.scale_min.get()
        max_db = self.scale_max.get()
        step = self.scale_step.get()
        
        if max_db <= min_db:
            max_db = min_db + 1
        if step <= 0:
            step = 2.5
        
        series_list = []
        mode = self.mode.get()
        
        if mode == "rms":
            if self.rms_data:
                angle = self.rms_data["angle_deg"]
                pol_type = self.rms_data.get("polarization", "linear")
                
                if pol_type == "circular":
                    for plane in ["YZ", "ZX", "XY"]:
                        for pol in ["RHCP", "LHCP", "RL"]:
                            key = f"{plane}_{pol}"
                            if key in self.rms_checkboxes and self.rms_checkboxes[key].get():
                                if self.rms_data[plane].get(pol) is not None:
                                    series_list.append(prepare_polar_series(
                                        angle, self.rms_data[plane][pol],
                                        min_db, max_db, f"{plane}_{pol}"
                                    ))
                else:
                    for plane in ["YZ", "ZX", "XY"]:
                        key = f"{plane}_HV"
                        if key in self.rms_checkboxes and self.rms_checkboxes[key].get():
                            if self.rms_data[plane].get("HV") is not None:
                                series_list.append(prepare_polar_series(
                                    angle, self.rms_data[plane]["HV"],
                                    min_db, max_db, f"{plane}_HV"
                                ))
            
            if self.pustelnik_data and "pustelnik" in self.rms_checkboxes:
                if self.rms_checkboxes["pustelnik"].get():
                    series_list.append(prepare_polar_series(
                        self.pustelnik_data["angle_deg"], self.pustelnik_data["HV"],
                        min_db, max_db, "Pustelnik"
                    ))
        
        elif mode == "theta_phi":
            for item in self.theta_phi_files:
                if item['var'].get():
                    series_list.append(prepare_polar_series(
                        item['theta'], item['db'],
                        min_db, max_db, item['name']
                    ))
        
        else:
            for item in self.combined_items:
                if item['var'].get():
                    series_list.append(prepare_polar_series(
                        item['angle'], item['values'],
                        min_db, max_db, item['name']
                    ))
        
        if not series_list:
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(series_list))))
        
        for i, series in enumerate(series_list):
            self.ax.plot(series['theta'], series['r'], 
                        color=colors[i % len(colors)], lw=2, label=series['label'])
        
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.set_rlim(0, max_db - min_db)
        
        ticks = []
        t = min_db
        while t <= max_db + 1e-9:
            ticks.append(round(t, 2))
            t += step
        
        self.ax.set_rticks([t - min_db for t in ticks])
        self.ax.set_yticklabels([f"{int(t)}" if abs(t-round(t))<1e-6 else f"{t:.1f}" for t in ticks])
        self.ax.grid(True, alpha=0.3)
        
        title = {"rms": "RMS/Circular", "theta_phi": "Theta/Phi", "combined": "Combined"}[mode]
        self.ax.set_title(title, pad=12)
        
        if series_list:
            self.ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
        
        self.canvas.draw_idle()
    
    def get_text_output(self):
        mode = self.mode.get()
        lines = [f"Polar mode: {mode}"]
        lines.append(f"Scale: {self.scale_min.get():.1f} to {self.scale_max.get():.1f} dB, step {self.scale_step.get():.1f}")
        
        if mode == "rms":
            if self.rms_data:
                lines.append(f"RMS type: {self.rms_data.get('polarization', 'linear')}")
        elif mode == "theta_phi":
            lines.append(f"Files loaded: {len(self.theta_phi_files)}")
        else:
            lines.append(f"Items: {len(self.combined_items)}")
        
        return "\n".join(lines)
