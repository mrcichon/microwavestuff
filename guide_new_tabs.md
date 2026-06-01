# Adding New Tabs - Quick Guide

## Files You Need (2 files)

### 1. `analysis_yourtab.py`
```python
import numpy as np

def compute_yourtab(files_data, param1, param2):
    """Your computation logic."""
    results = {}
    for entry in files_data:
        # Do your math here
        results[entry['name']] = some_calculation(entry['signal'])
    return {'results': results, 'param1': param1, 'param2': param2}

def format_yourtab_text(result_dict):
    """Format results as text."""
    lines = []
    lines.append("Your Analysis Results")
    lines.append("-" * 40)
    for name, value in result_dict['results'].items():
        lines.append(f"{name}: {value:.3f}")
    return "\n".join(lines)
```

### 2. `ui_tab_yourtab.py`
```python
import tkinter as tk
from tkinter import ttk
from analysis_yourtab import compute_yourtab, format_yourtab_text

class TabYourTab:
    def __init__(self, parent, fig, ax, canvas, get_files_func, get_freq_range_func):
        self.parent = parent
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.get_files = get_files_func
        self.get_freq_range = get_freq_range_func
        
        # Your UI variables
        self.param1_var = tk.DoubleVar(value=1.0)
        
        self.last_result = None
        self._build_ui()
    
    def _build_ui(self):
        # Horizontal control bar at top
        frame = ttk.Frame(self.parent)
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame, text="Param1:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Entry(frame, textvariable=self.param1_var, width=6).pack(side=tk.LEFT)
        
        ttk.Button(frame, text="Update", command=self.update).pack(side=tk.LEFT, padx=10)
        
        self.control_panel = frame
    
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
                # Load network (with caching)
                ntw_full = d.get('ntwk_full')
                if ntw_full is None:
                    from sparams_io import loadFile
                    ntw_full = loadFile(p)
                    d['ntwk_full'] = ntw_full
                
                # Check cache
                if d.get('cached_range') != sstr:
                    ntw = ntw_full[sstr]
                    d['ntwk'] = ntw
                    d['cached_range'] = sstr
                else:
                    ntw = d['ntwk']
                
                fname = d.get('custom_name') if d.get('is_average') else Path(p).stem
                
                # Extract what you need
                signal = ntw.s21.s_db.flatten()  # or whatever param
                
                files_data.append({
                    'name': fname,
                    'signal': signal,
                    'freq': ntw.f
                })
                
            except Exception:
                pass
        
        return files_data
    
    def update(self):
        files_data = self._get_files_data()
        
        if not files_data:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No files selected", 
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            self.last_result = None
            return
        
        # Compute
        result = compute_yourtab(files_data, self.param1_var.get(), "whatever")
        self.last_result = result
        
        # Plot
        self.ax.clear()
        # Your matplotlib plotting code
        self.ax.bar(list(result['results'].keys()), list(result['results'].values()))
        self.ax.set_xlabel('Files')
        self.ax.set_ylabel('Value')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def get_text_output(self):
        if self.last_result is None:
            return ""
        return format_yourtab_text(self.last_result)
```

## Register in `ui_main.py`

### Add creation method:
```python
def _create_yourtab_tab(self):
    frame = ttk.Frame(self.nb)
    self.nb.add(frame, text="Your Tab")
    
    self.figYT, self.axYT = plt.subplots(figsize=(10, 8))
    self.cvYT = FigureCanvasTkAgg(self.figYT, master=frame)
    self.cvYT.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    self.tbYT = NavigationToolbar2Tk(self.cvYT, frame)
    self.tbYT.update()
    self.tbYT.pack(fill=tk.X)
    
    from ui_tab_yourtab import TabYourTab
    self.tab_yourtab = TabYourTab(
        parent=frame,
        fig=self.figYT,
        ax=self.axYT,
        canvas=self.cvYT,
        get_files_func=self.get_files,
        get_freq_range_func=self.get_freq_range
    )
```

### Call it in `_makeUi()`:
```python
self._create_yourtab_tab()
```

### Add to tab mapping:
```python
def _get_tab_by_name(self, name):
    tab_map = {
        # ... existing tabs ...
        "Your Tab": self.tab_yourtab,
    }
    return tab_map.get(name)
```

## Notes

1. **Controls go horizontal**: `frame.pack(side=tk.TOP, fill=tk.X)`, controls pack with `side=tk.LEFT`
2. **Always cache networks**: Check `d['cached_range']` against current `sstr`
3. **Store last result**: For `get_text_output()` to work
4. **Handle empty case**: Show "No files selected" message
5. **Use callbacks**: Never access `App` directly, use `self.get_files()` and `self.get_freq_range()`

## Using sparams_io Functions

### Available Loaders in sparams_io.py

```python
from sparams_io import loadFile  # TouchStone files (.s1p, .s2p, .s3p)
from sparams_io import parse_megiq_txt  # MegiQ antenna patterns
from sparams_io import parse_pustelnik_txt  # Pustelnik format
from sparams_io import parse_theta_phi_file  # Theta-phi patterns

# loadFile() handles encoding issues and comma/decimal conversion automatically
# Returns: rf.Network object

# parse_megiq_txt() returns:
# {'angle_deg': array, 'YZ': {'HV': array}, 'ZX': {'HV': array}, 'XY': {'HV': array}}

# parse_pustelnik_txt() returns:
# {'angle_deg': array, 'HV': array}

# parse_theta_phi_file() returns:
# DataFrame with columns: ['theta_deg', 'phi_deg', 'dir_dbi']
```

### Standard Pattern

```python
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
            # Check cache first
            ntw_full = d.get('ntwk_full')
            if ntw_full is None:
                from sparams_io import loadFile  # <-- THE MAGIC
                ntw_full = loadFile(p)
                d['ntwk_full'] = ntw_full
            
            # Slice to current range
            if d.get('cached_range') != sstr:
                ntw = ntw_full[sstr]
                d['ntwk'] = ntw
                d['cached_range'] = sstr
            else:
                ntw = d['ntwk']
            
            fname = d.get('custom_name') if d.get('is_average') else Path(p).stem
            
            # Get what you need from network
            signal = ntw.s21.s_db.flatten()
            
            files_data.append({
                'name': fname,
                'signal': signal,
                'freq': ntw.f
            })
            
        except Exception:
            pass
    
    return files_data
```

```python
# polar shit
from sparams_io import parse_megiq_txt, parse_pustelnik_txt

def _get_files_data(self):
    files_data = []
    
    for v, p, d in self.get_files():
        if not v.get():
            continue
        
        fname = Path(p).stem
        
        # Detect format by filename
        if 'megiq' in p.lower():
            try:
                data = parse_megiq_txt(p)
                # data['angle_deg'] - angles
                # data['YZ']['HV'] - YZ plane data
                files_data.append({
                    'name': fname,
                    'type': 'megiq',
                    'data': data
                })
            except Exception:
                pass
                
        elif 'pustelnik' in p.lower():
            try:
                data = parse_pustelnik_txt(p)
                # data['angle_deg'] - angles
                # data['HV'] - gain values
                files_data.append({
                    'name': fname,
                    'type': 'pustelnik',
                    'data': data
                })
            except Exception:
                pass
    
    return files_data
```

### S-Parameter Quick Reference

```python
# What you get from loadFile():
ntw = loadFile(path)  # rf.Network object

# Frequency info
ntw.f  # Frequency array in Hz
ntw.frequency.f_scaled  # In current units
ntw.nports  # Number of ports (1 or 2)

# S-parameters (for 2-port):
ntw.s11, ntw.s12, ntw.s21, ntw.s22

# Each S-param has:
.s_db        # Magnitude in dB
.s_mag       # Linear magnitude  
.s_deg       # Phase in degrees
.s_rad       # Phase in radians
.s_time_db   # Time domain in dB
.frequency.t_ns  # Time array in ns

# Always flatten for plotting:
values = ntw.s21.s_db.flatten()
```

### Caching Reminder

```python
# The pattern is always:
# 1. Check if loaded: d.get('ntwk_full')
# 2. If not, load it: loadFile(p)
# 3. Store it: d['ntwk_full'] = ntw_full
# 4. Check if sliced for current range: d.get('cached_range') != sstr
# 5. If not, slice it: ntw_full[sstr]
# 6. Store slice: d['ntwk'] = ntw

# This means files are loaded ONCE, sliced as needed
```

## That's it. Copy, paste, modify.
