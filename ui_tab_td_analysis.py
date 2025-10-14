import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pathlib import Path

from analysis_td_peaks import find_td_peaks, format_td_analysis_text

class TabTDAnalysis:
    def __init__(self, parent, fig, canvas,
                 get_files_func, get_freq_range_func):
        self.parent = parent
        self.fig = fig
        self.canvas = canvas
        self.ax1 = None
        self.ax2 = None
        self.get_files = get_files_func
        self.get_freq_range = get_freq_range_func
        
        self.td_time_limit = tk.DoubleVar(value=50.0)
        self.td_mark_peaks = tk.BooleanVar(value=True)
        
        self.td_analysis_data = None
        self.td_markers = []
        self.td_marker_text = []
        
        self._build_ui()
        
    def _build_ui(self):
        frame = ttk.Frame(self.parent)
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame, text="Time limit [ns]:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Entry(frame, textvariable=self.td_time_limit,
                 width=6).pack(side=tk.LEFT, padx=2)
        
        ttk.Checkbutton(frame, text="Mark peaks",
                       variable=self.td_mark_peaks,
                       command=self.update).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(frame, text="Export Analysis",
                  command=self._export_analysis).pack(side=tk.LEFT, padx=10)
        ttk.Button(frame, text="Clear Markers",
                  command=self._clear_markers).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(frame, text="Checks: t_max(S11) ≈ 2×t_max(S21)",
                 font=("", 8), foreground="gray").pack(side=tk.LEFT, padx=10)
        
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
                    
                    # Get S11 time domain
                    s11_td = ntw.s11
                    t_ns = s11_td.frequency.t_ns
                    s11_time_db = s11_td.s_time_db.flatten()
                    
                    # Get S21 if available
                    s21_time_data = None
                    if ntw.nports >= 2:
                        s21_td = ntw.s21
                        s21_time_db = s21_td.s_time_db.flatten()
                        s21_time_data = {'t_ns': t_ns, 'values': s21_time_db}
                    
                    files_data.append({
                        'name': fname,
                        's11_time': {'t_ns': t_ns, 'values': s11_time_db},
                        's21_time': s21_time_data,
                        'color': d.get('line_color'),
                        'linewidth': d.get('line_width', 1.0)
                    })
                    
                except Exception:
                    pass
        
        return files_data
    
    def update(self):
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(211, sharex=None)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        
        files_data = self._get_files_data()
        
        if not files_data:
            self.ax1.text(0.5, 0.5, "No data available",
                         ha="center", va="center", fontsize=12, color="gray")
            self.ax1.set_xticks([])
            self.ax1.set_yticks([])
            self.ax2.set_visible(False)
            self.canvas.draw()
            self.td_analysis_data = None
            return
        
        # Analyze peaks
        time_limit = self.td_time_limit.get()
        self.td_analysis_data = find_td_peaks(files_data, time_limit)
        
        # Plot S11
        for file_data, analysis in zip(files_data, self.td_analysis_data):
            t_ns = file_data['s11_time']['t_ns']
            s11_td = file_data['s11_time']['values']
            
            # Apply time limit
            time_mask = (t_ns >= 0) & (t_ns <= time_limit)
            t_limited = t_ns[time_mask]
            s11_limited = s11_td[time_mask]
            
            kwargs = {'label': file_data['name']}
            if file_data['color']:
                kwargs['color'] = file_data['color']
            if file_data['linewidth'] != 1.0:
                kwargs['linewidth'] = file_data['linewidth']
            
            line = self.ax1.plot(t_limited, s11_limited, **kwargs)[0]
            
            # Mark peak if requested
            if self.td_mark_peaks.get():
                if analysis['s21_max_time'] is not None:
                    color = 'green' if analysis['condition_met'] else 'red'
                else:
                    color = 'gray'
                
                self.ax1.plot(analysis['s11_max_time'], analysis['s11_max_val'],
                             'o', color=color, markersize=8,
                             markeredgecolor='black', markeredgewidth=1)
        
        self.ax1.set_ylabel('S11 [dB]')
        self.ax1.set_title('S11 Time Domain')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(0, time_limit)
        self.ax1.legend(loc='best', fontsize=8)
        
        # Plot S21
        has_s21 = False
        for file_data, analysis in zip(files_data, self.td_analysis_data):
            if file_data['s21_time'] is not None:
                t_ns = file_data['s21_time']['t_ns']
                s21_td = file_data['s21_time']['values']
                
                time_mask = (t_ns >= 0) & (t_ns <= time_limit)
                t_limited = t_ns[time_mask]
                s21_limited = s21_td[time_mask]
                
                kwargs = {'label': file_data['name']}
                if file_data['color']:
                    kwargs['color'] = file_data['color']
                if file_data['linewidth'] != 1.0:
                    kwargs['linewidth'] = file_data['linewidth']
                
                self.ax2.plot(t_limited, s21_limited, **kwargs)
                has_s21 = True
                
                # Mark peak if requested
                if self.td_mark_peaks.get() and analysis['s21_max_time'] is not None:
                    color = 'green' if analysis['condition_met'] else 'red'
                    self.ax2.plot(analysis['s21_max_time'], analysis['s21_max_val'],
                                 'o', color=color, markersize=8,
                                 markeredgecolor='black', markeredgewidth=1)
        
        if has_s21:
            self.ax2.set_ylabel('S21 [dB]')
            self.ax2.set_xlabel('Time [ns]')
            self.ax2.set_title('S21 Time Domain')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_xlim(0, time_limit)
            self.ax2.legend(loc='best', fontsize=8)
        else:
            self.ax2.text(0.5, 0.5, "No S21 data available (1-port devices)",
                         ha="center", va="center", color="gray", transform=self.ax2.transAxes)
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _clear_markers(self):
        for m in self.td_markers:
            try:
                m['marker'].remove()
                m['text'].remove()
            except:
                pass
        self.td_markers.clear()
        self.td_marker_text.clear()
        self.update()
    
    def _export_analysis(self):
        if not self.td_analysis_data:
            messagebox.showinfo("No data", "No analysis data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save TD Analysis",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        if filename.endswith('.csv'):
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['File', 'S11_max_time_ns', 'S11_max_val_dB',
                               'S21_max_time_ns', 'S21_max_val_dB', 'Condition_met'])
                for res in self.td_analysis_data:
                    writer.writerow([
                        res['file'],
                        res['s11_max_time'],
                        res['s11_max_val'],
                        res['s21_max_time'] if res['s21_max_time'] is not None else 'N/A',
                        res['s21_max_val'] if res['s21_max_val'] is not None else 'N/A',
                        'PASS' if res['condition_met'] else 'FAIL' if res['condition_met'] is not None else 'N/A'
                    ])
        else:
            with open(filename, 'w') as f:
                f.write(format_td_analysis_text(self.td_analysis_data, self.td_time_limit.get()))
        
        messagebox.showinfo("Export complete", f"Analysis saved to {filename}")
    
    def get_text_output(self):
        if self.td_analysis_data is None:
            return ""
        return format_td_analysis_text(self.td_analysis_data, self.td_time_limit.get())
