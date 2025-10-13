import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import io, math, re, os
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

# ===================== SKALA SZTYWNA PER ZAKŁADKA =====================
RMS_SCALE = (-10.0, 10.0, 2.5)   # min, max, step dla zakładki RMS + Pustelnik
TP_SCALE  = (-40.0, 20.0, 20.0)  # min, max, step dla zakładki Theta/Phi

# ===================== PARSERY =====================
def parse_megiq_txt(path):
    """
    Parsuje pliki MegiQ RMS - obsługuje zarówno polaryzację liniową (HV) 
    jak i kołową (RHCP/LHCP)
    """
    print(f"\n=== DEBUG: Parsing file: {path} ===")
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    
    print(f"File size: {len(raw)} bytes")
    
    raw = raw.lstrip("\ufeff")
    text = raw.replace("\r\n", "\n").replace(",", ".")
    lines = text.split("\n")
    
    print(f"Total lines: {len(lines)}")
    print(f"First 10 lines preview:")
    for i, line in enumerate(lines[:10]):
        print(f"  Line {i}: {line[:100]}")

    header_idx = None
    is_circular = False
    
    # Sprawdź czy to polaryzacja kołowa (RHCP/LHCP)
    print("\nSearching for RHCP/LHCP header...")
    for i, line in enumerate(lines):
        if "RHCP" in line and "LHCP" in line:
            print(f"  Found RHCP/LHCP at line {i}: {line[:150]}")
            if "dB.YZ" in line or "dB.ZX" in line or "dB.XY" in line:
                header_idx = i
                is_circular = True
                print(f"  ✓ Confirmed as circular polarization header")
                break
    
    # Jeśli nie znaleziono kołowej, sprawdź liniową (HV)
    if header_idx is None:
        print("\nSearching for HV header...")
        for i, line in enumerate(lines):
            if "dB.YZ.H" in line and "dB.ZX.H" in line and "dB.XY.H" in line:
                print(f"  Found HV at line {i}: {line[:150]}")
                header_idx = i
                is_circular = False
                print(f"  ✓ Confirmed as linear polarization header")
                break
    
    if header_idx is None:
        print("\n✗ ERROR: No valid header found!")
        print("Searched for patterns:")
        print("  - Circular: 'RHCP' + 'LHCP' + 'dB.YZ'")
        print("  - Linear: 'dB.YZ.H' + 'dB.ZX.H' + 'dB.XY.H'")
        raise ValueError("Nie znaleziono nagłówka z kolumnami dB.* (ani HV ani RHCP/LHCP)")

    header_line = lines[header_idx]
    hdr_tokens = [t for t in header_line.split("\t")]
    colnames = ["angle_deg"] + [t for t in hdr_tokens[1:] if t]
    
    print(f"\nParsed column names: {colnames}")

    data_block = "\n".join(lines[header_idx + 1:])
    print(f"Data block starts at line {header_idx + 1}, contains {len(lines[header_idx + 1:])} lines")
    print(f"First data line: {lines[header_idx + 1][:100] if header_idx + 1 < len(lines) else 'N/A'}")
    
    df = pd.read_csv(io.StringIO(data_block), sep=r"\s+|\t+", engine="python", names=colnames)
    
    print(f"DataFrame shape after parsing: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"First few rows:\n{df.head()}")

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["angle_deg"]).reset_index(drop=True)
    
    print(f"DataFrame shape after cleaning: {df.shape}")

    if is_circular:
        # Polaryzacja kołowa - oblicz także R+L dla każdej płaszczyzny
        print("\n✓ Returning CIRCULAR polarization data")
        
        def calc_rl_sum(rhcp_series, lhcp_series):
            """Oblicza sumę mocy R+L w dB"""
            if rhcp_series is None or lhcp_series is None:
                return None
            rhcp_linear = 10**(rhcp_series / 10.0)
            lhcp_linear = 10**(lhcp_series / 10.0)
            rl_power = rhcp_linear + lhcp_linear
            rl_db = 10 * np.log10(rl_power)
            return rl_db
        
        result = {
            "angle_deg": df["angle_deg"],
            "polarization": "circular",
            "YZ": {
                "RHCP": df.get("dB.YZ.RHCP"),
                "LHCP": df.get("dB.YZ.LHCP"),
                "RL": calc_rl_sum(df.get("dB.YZ.RHCP"), df.get("dB.YZ.LHCP"))
            },
            "ZX": {
                "RHCP": df.get("dB.ZX.RHCP"),
                "LHCP": df.get("dB.ZX.LHCP"),
                "RL": calc_rl_sum(df.get("dB.ZX.RHCP"), df.get("dB.ZX.LHCP"))
            },
            "XY": {
                "RHCP": df.get("dB.XY.RHCP"),
                "LHCP": df.get("dB.XY.LHCP"),
                "RL": calc_rl_sum(df.get("dB.XY.RHCP"), df.get("dB.XY.LHCP"))
            },
        }
        print(f"  YZ.RHCP: {'Found' if result['YZ']['RHCP'] is not None else 'Missing'}")
        print(f"  YZ.LHCP: {'Found' if result['YZ']['LHCP'] is not None else 'Missing'}")
        print(f"  YZ.R+L:  {'Calculated' if result['YZ']['RL'] is not None else 'Missing'}")
        print(f"  ZX.RHCP: {'Found' if result['ZX']['RHCP'] is not None else 'Missing'}")
        print(f"  ZX.LHCP: {'Found' if result['ZX']['LHCP'] is not None else 'Missing'}")
        print(f"  ZX.R+L:  {'Calculated' if result['ZX']['RL'] is not None else 'Missing'}")
        print(f"  XY.RHCP: {'Found' if result['XY']['RHCP'] is not None else 'Missing'}")
        print(f"  XY.LHCP: {'Found' if result['XY']['LHCP'] is not None else 'Missing'}")
        print(f"  XY.R+L:  {'Calculated' if result['XY']['RL'] is not None else 'Missing'}")
        return result
    else:
        # Polaryzacja liniowa (HV)
        print("\n✓ Returning LINEAR polarization data")
        result = {
            "angle_deg": df["angle_deg"],
            "polarization": "linear",
            "YZ": {"HV": df.get("dB.YZ.HV")},
            "ZX": {"HV": df.get("dB.ZX.HV")},
            "XY": {"HV": df.get("dB.XY.HV")},
        }
        print(f"  YZ.HV: {'Found' if result['YZ']['HV'] is not None else 'Missing'}")
        print(f"  ZX.HV: {'Found' if result['ZX']['HV'] is not None else 'Missing'}")
        print(f"  XY.HV: {'Found' if result['XY']['HV'] is not None else 'Missing'}")
        return result

def parse_pustelnik_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    raw = raw.lstrip("\ufeff")
    text = raw.replace("\r\n", "\n").replace(",", ".")
    lines = text.split("\n")

    header_idx = None
    hv_header_regex = re.compile(r"HV", re.I)
    for i, line in enumerate(lines):
        if hv_header_regex.search(line) and ("dB." in line or "dB" in line):
            header_idx = i
            break

    if header_idx is not None:
        header_line = lines[header_idx]
        hdr_tokens = [t for t in header_line.split("\t")]
        colnames = ["angle_deg"] + [t for t in hdr_tokens[1:] if t]
        data_block = "\n".join(lines[header_idx + 1:])
        df = pd.read_csv(io.StringIO(data_block), sep=r"\s+|\t+", engine="python", names=colnames)
        hv_col = None
        for c in df.columns:
            if re.search(r"HV$", c, re.I):
                hv_col = c; break
        if hv_col is None and "HV" in df.columns: hv_col = "HV"
        if hv_col is None and len(df.columns) >= 2: hv_col = df.columns[1]
        df = df.dropna(subset=["angle_deg"]).reset_index(drop=True)
        hv = pd.to_numeric(df[hv_col], errors="coerce")
        return {"angle_deg": pd.to_numeric(df["angle_deg"], errors="coerce"), "HV": hv}
    else:
        # minimalny format: kąt, HV
        start = 0
        number_line = re.compile(r"^\s*-?\d+(\.\d+)?(\s+|-?\d|\.)+")
        for i, ln in enumerate(lines):
            if number_line.match(ln.strip()):
                start = i; break
        block = "\n".join(l for l in lines[start:] if l.strip())
        df = pd.read_csv(io.StringIO(block), sep=r"\s+|\t+", engine="python", header=None)
        if df.shape[1] < 2:
            raise ValueError("Plik pustelnik musi mieć co najmniej 2 kolumny: kąt i HV")
        df = df.iloc[:, :2]
        df.columns = ["angle_deg", "HV"]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["angle_deg", "HV"]).reset_index(drop=True)
        return {"angle_deg": df["angle_deg"], "HV": df["HV"]}

def parse_theta_phi_file(path):
    """
    Pliki typu: Theta [deg], Phi [deg], Abs(Dir.)[dBi], ...
    UWAGA: nagłówek ma spacje w nawiasach -> nie parsujemy po nazwach,
    tylko bierzemy 3 pierwsze wartości liczbowe z każdego wiersza danych:
    Theta, Phi, Abs(Dir.)[dBi].
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    raw = raw.lstrip("\ufeff").replace("\r\n", "\n").replace(",", ".")
    lines = [ln for ln in raw.split("\n")
             if ln.strip() and not set(ln.strip()) <= set("- ")]

    # znajdź linię nagłówka, ale jej nie parsujemy – tylko pomijamy
    header_i = None
    for i, ln in enumerate(lines):
        if "Theta" in ln and "Phi" in ln:
            header_i = i
            break
    data_lines = lines[header_i + 1:] if header_i is not None else lines

    # wyciągaj liczby z każdej linii; weź 3 pierwsze (Theta, Phi, Abs(Dir.)[dBi])
    float_re = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
    rows = []
    for ln in data_lines:
        nums = [float(x) for x in float_re.findall(ln)]
        if len(nums) >= 3:
            rows.append((nums[0], nums[1], nums[2]))

    if not rows:
        raise ValueError("Nie znalazłem danych liczbowych w pliku.")

    df = pd.DataFrame(rows, columns=["theta_deg", "phi_deg", "dir_dbi"])
    return df


# ===== auto-wybór i łączenie przekrojów Theta/Phi (bez UI φ) =====
def _nearest_mod180(phivals, p, tol=2.0):
    p2 = (p + 180.0) % 360.0
    diffs = np.minimum(np.abs(phivals - p2), 360.0 - np.abs(phivals - p2))
    idx = np.where(diffs <= tol)[0]
    return (int(idx[0]) if idx.size else None)

def auto_extract_tp_series(df, tol=2.0):
    """Zwraca (theta, dB, opis_phi). Gdy są dwa przekroje ~180°, składa je w 0..360;
       gdy jeden – domyka symetrycznie 360−θ jeśli potrzeba."""
    phivals = np.array(sorted(np.round(df["phi_deg"].to_numpy(), 1)))
    uniq = np.unique(phivals)
    if uniq.size == 0:
        raise ValueError("Brak wartości φ w pliku.")

    counts = [(phi, (phivals == phi).sum()) for phi in uniq]
    counts.sort(key=lambda x: x[1], reverse=True)
    phi_a = counts[0][0]
    j = _nearest_mod180(uniq, phi_a, tol=tol)

    if j is not None:
        phi_b = float(uniq[j])
        a = df[np.isclose(df["phi_deg"], phi_a, atol=tol)].sort_values("theta_deg")
        b = df[np.isclose(df["phi_deg"], phi_b, atol=tol)].sort_values("theta_deg", ascending=False)
        theta = np.concatenate([a["theta_deg"].to_numpy(), (360.0 - b["theta_deg"].to_numpy())])
        db    = np.concatenate([a["dir_dbi"].to_numpy(),   b["dir_dbi"].to_numpy()])
        if theta.size and abs(theta[-1]-360.0) < 1e-6:
            theta, db = theta[:-1], db[:-1]
        used = f"φ≈{phi_a}° & {phi_b}°"
    else:
        a = df[np.isclose(df["phi_deg"], phi_a, atol=tol)].sort_values("theta_deg")
        theta = a["theta_deg"].to_numpy()
        db    = a["dir_dbi"].to_numpy()
        span = float(theta.max() - theta.min()) if theta.size else 0.0
        if span < 300:
            theta2 = 360.0 - theta[::-1]
            db2    = db[::-1]
            if theta2.size and abs(theta2[0] - 360.0) < 1e-6:
                theta2, db2 = theta2[1:], db2[1:]
            theta = np.concatenate([theta, theta2])
            db    = np.concatenate([db, db2])
        used = f"φ≈{phi_a}°"
    
    # Close the loop: add first point at the end (0° = 360°)
    if theta.size > 0:
        # Check if we need to close (not already at 360°)
        if abs(theta[-1] - 360.0) > 1e-3:
            theta = np.append(theta, 360.0)
            db = np.append(db, db[0])
        # Or if starting from non-zero, prepend 0°
        elif abs(theta[0]) > 1e-3:
            theta = np.insert(theta, 0, 0.0)
            db = np.insert(db, 0, db[-1])
    
    return theta, db, used

# ===================== KOLORY =====================
# Kolory dla płaszczyzn (linear HV)
HV_COLOR = {
    "xawery":    "#e91e63",
    "zuzia":     "#2196f3",
    "yeti":      "#43a047",
    "pustelnik": "#ffd400",
}

# Kolory dla polaryzacji kołowej (circular)
CIRCULAR_COLOR = {
    "xawery_RHCP": "#e91e63",
    "xawery_LHCP": "#f06292",  # jaśniejszy różowy
    "xawery_RL":   "#880e4f",  # ciemniejszy różowy dla R+L
    "zuzia_RHCP":  "#2196f3",
    "zuzia_LHCP":  "#64b5f6",  # jaśniejszy niebieski
    "zuzia_RL":    "#0d47a1",  # ciemniejszy niebieski dla R+L
    "yeti_RHCP":   "#43a047",
    "yeti_LHCP":   "#81c784",  # jaśniejszy zielony
    "yeti_RL":     "#1b5e20",  # ciemniejszy zielony dla R+L
}

# ===================== RYSOWANIE =====================
def draw_polar(ax, series_list, title_text, min_db, max_db, grid_step):
    """series_list: [{'label','angle_deg','val','color'}, ...]"""
    print(f"\n=== DRAW_POLAR called ===")
    print(f"  Title: {title_text}")
    print(f"  Scale: {min_db} to {max_db}, step {grid_step}")
    print(f"  Number of series: {len(series_list)}")
    
    ax.clear()
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")

    if max_db <= min_db: max_db = min_db + 1e-3
    if grid_step <= 0: grid_step = 2.5

    offset = -min_db
    to_r = lambda v: v + offset

    for i, s in enumerate(series_list):
        print(f"  Series {i}: {s.get('label', 'unnamed')}, color={s.get('color', 'N/A')}")
        theta = pd.Series(s["angle_deg"]).to_numpy() * math.pi / 180.0
        vals = pd.Series(s["val"]).astype(float).to_numpy()
        print(f"    Theta range: {theta.min():.2f} to {theta.max():.2f} rad ({len(theta)} points)")
        print(f"    Values range: {vals.min():.2f} to {vals.max():.2f} dB ({len(vals)} points)")
        vals = np.clip(vals, min_db, max_db)
        ax.plot(theta, to_r(vals), color=s["color"], lw=2.2, label=s.get("label",""))
        print(f"    ✓ Plotted")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, to_r(max_db))

    ticks, t = [], min_db
    while t <= max_db + 1e-9:
        ticks.append(round(t, 2)); t += grid_step
    ax.set_rticks([to_r(x) for x in ticks])
    ax.set_yticklabels([f"{int(x)}" if abs(x-round(x))<1e-6 else f"{x:.1f}"
                        for x in ticks], color="#444")
    ax.set_xticklabels([f"{d}°" for d in range(0, 360, 45)], color="#444")
    ax.grid(True, color="#cccccc", alpha=0.9)
    ax.set_title(title_text, color="#111", pad=12, fontsize=12)
    print("  ✓ draw_polar complete")

# ===================== BOX-ZOOM (CTRL + drag) =====================
class DebugRedirect:
    """Redirects stdout to a tkinter Text widget"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = []
        
    def write(self, message):
        if message.strip():  # Only log non-empty messages
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.insert(tk.END, message)
            # Keep only last 100 lines
            lines = int(self.text_widget.index('end-1c').split('.')[0])
            if lines > 150:
                self.text_widget.delete('1.0', f'{lines-100}.0')
            self.text_widget.see(tk.END)
            self.text_widget.config(state=tk.DISABLED)
            self.text_widget.update_idletasks()
    
    def flush(self):
        pass

class BoxZoom:
    def __init__(self, ax, canvas):
        self.ax = ax
        self.canvas = canvas
        self.start = None
        self.rect = None
        self.cid_press = canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_move  = canvas.mpl_connect('motion_notify_event', self.on_move)
        self.cid_rel   = canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1 or not self._ctrl(event): return
        if event.xdata is None or event.ydata is None: return
        self.start = (event.xdata, event.ydata)
        self.rect = Rectangle((self.start[0], self.start[1]), 0, 0,
                              fill=False, ec='#666', lw=1.2, ls='--')
        self.ax.add_patch(self.rect); self.canvas.draw_idle()

    def on_move(self, event):
        if self.start is None or self.rect is None or event.inaxes != self.ax: return
        if event.xdata is None or event.ydata is None: return
        x0, y0 = self.start
        self.rect.set_width(event.xdata - x0)
        self.rect.set_height(event.ydata - y0)
        self.rect.set_xy((x0, y0))
        self.canvas.draw_idle()

    def on_release(self, event):
        if self.start is None or self.rect is None or event.inaxes != self.ax: self._cleanup(); return
        if event.xdata is None or event.ydata is None: self._cleanup(); return
        x0, y0 = self.start
        x1, y1 = event.xdata, event.ydata
        rmin, rmax = sorted([y0, y1])
        t0, t1 = x0, x1
        twopi = 2*np.pi
        t0 = (t0 + twopi) % twopi
        t1 = (t1 + twopi) % twopi
        if (t1 - t0) % twopi > np.pi: t0, t1 = t1, t0
        if t1 <= t0: t1 += twopi
        try: self.ax.set_xlim(t0, t1)
        except Exception: pass
        self.ax.set_rlim(rmin, rmax)
        self.canvas.draw_idle()
        self._cleanup()

    def _cleanup(self):
        self.start = None
        if self.rect is not None:
            self.rect.remove()
            self.rect = None
        self.canvas.draw_idle()

    @staticmethod
    def _ctrl(event):
        return event.key is not None and ('control' in event.key or 'ctrl' in event.key)

# ===================== ZAKŁADKA 1 – RMS + Pustelnik =====================
class TabRMS(ttk.Frame):
    def __init__(self, master, ax, canvas):
        super().__init__(master)
        self.ax, self.canvas = ax, canvas

        left = tk.Frame(self, bg="#f5f5f5"); left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        tk.Label(left, text="Plik TXT (MegiQ RMS):", bg="#f5f5f5",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.path_var = tk.StringVar(value="")
        row = tk.Frame(left, bg="#f5f5f5"); row.pack(fill=tk.X, pady=6)
        tk.Entry(row, textvariable=self.path_var, width=45).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(row, text="Wybierz plik…", command=self.pick_file_main).pack(side=tk.LEFT)

        # Ramka na checkboxy - będzie dynamicznie przebudowywana
        self.checkboxes_frame = tk.Frame(left, bg="#f5f5f5")
        self.checkboxes_frame.pack(anchor="w", fill=tk.X, pady=(10,0))

        # Pustelnik
        tk.Label(left, text="Pustelnik (osobny plik, tylko HV):", bg="#f5f5f5",
                 font=("Segoe UI",10,"bold")).pack(anchor="w", pady=(12,2))
        self.pust_path_var = tk.StringVar(value="")
        rowp = tk.Frame(left, bg="#f5f5f5"); rowp.pack(fill=tk.X, pady=4)
        tk.Entry(rowp, textvariable=self.pust_path_var, width=45).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(rowp, text="Wybierz pustelnika…", command=self.pick_file_pust).pack(side=tk.LEFT)
        self.var_pust = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="pustelnik (HV, żółty)", variable=self.var_pust, command=self.update_plot).pack(anchor="w")

        ttk.Button(left, text="Wczytaj i narysuj", command=self.load_files).pack(anchor="w", pady=(12,0))
        
        # Status/debug area
        self.status_label = tk.Label(left, text="Status: Czekam na plik...", bg="#f5f5f5", fg="#666", 
                                     wraplength=400, justify="left", font=("Segoe UI", 9))
        self.status_label.pack(anchor="w", pady=(8,0))
        
        tk.Label(left, text="Legenda:", bg="#f5f5f5",
                 font=("Segoe UI",10,"bold")).pack(anchor="w")
        self.legend_frame = tk.Frame(left, bg="#f5f5f5")
        self.legend_frame.pack(anchor="w", fill=tk.X, pady=(2,8))

        tk.Label(left, text="Zoom: CTRL+przeciągnij (box), kółko – promień, Home – reset",
                 bg="#f5f5f5", fg="#444", justify="left").pack(anchor="w", pady=(6,0))
        
        # Debug console
        debug_header = tk.Frame(left, bg="#f5f5f5")
        debug_header.pack(anchor="w", fill=tk.X, pady=(10,0))
        tk.Label(debug_header, text="Debug Log:", bg="#f5f5f5", 
                 font=("Segoe UI",10,"bold")).pack(side=tk.LEFT)
        ttk.Button(debug_header, text="Wyczyść log", 
                   command=lambda: self._clear_debug()).pack(side=tk.RIGHT)
        
        debug_frame = tk.Frame(left, bg="#f5f5f5", relief=tk.SUNKEN, bd=1)
        debug_frame.pack(anchor="w", fill=tk.BOTH, expand=True, pady=(2,0))
        
        self.debug_text = tk.Text(debug_frame, height=10, width=50, bg="#2b2b2b", fg="#00ff00",
                                  font=("Consolas", 8), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(debug_frame, command=self.debug_text.yview)
        self.debug_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.debug_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.debug_text.config(state=tk.DISABLED)

        self.data_main = None
        self.data_pust = None
        self.polarization_type = None  # "linear" lub "circular"
        
        # Zmienne dla checkboxów - będą tworzone dynamicznie
        self.checkbox_vars = {}
        
        self.update_legend([])
        self._build_initial_checkboxes()
        
        # Redirect stdout to debug widget
        import sys
        self.original_stdout = sys.stdout
        sys.stdout = DebugRedirect(self.debug_text)

    def debug_log(self, message):
        """Add message to debug text widget"""
        self.debug_text.config(state=tk.NORMAL)
        self.debug_text.insert(tk.END, message + "\n")
        # Keep only last 100 lines
        lines = int(self.debug_text.index('end-1c').split('.')[0])
        if lines > 100:
            self.debug_text.delete('1.0', f'{lines-100}.0')
        self.debug_text.see(tk.END)
        self.debug_text.config(state=tk.DISABLED)
        self.update_idletasks()
    
    def _clear_debug(self):
        """Clear the debug log"""
        self.debug_text.config(state=tk.NORMAL)
        self.debug_text.delete('1.0', tk.END)
        self.debug_text.config(state=tk.DISABLED)
        print("Debug log cleared")

    def _build_initial_checkboxes(self):
        """Tworzy początkowy zestaw checkboxów (dla HV)"""
        print("Building initial checkboxes...")
        for w in self.checkboxes_frame.winfo_children():
            w.destroy()
        
        tk.Label(self.checkboxes_frame, text="Pokaż płaszczyzny:", bg="#f5f5f5",
                 font=("Segoe UI",10,"bold")).pack(anchor="w", pady=(0,2))
        
        self.checkbox_vars = {
            "xawery": tk.BooleanVar(value=True),
            "yeti": tk.BooleanVar(value=True),
            "zuzia": tk.BooleanVar(value=True)
        }
        
        ttk.Checkbutton(self.checkboxes_frame, text="xawery (YZ) HV", 
                       variable=self.checkbox_vars["xawery"], 
                       command=self.update_plot).pack(anchor="w")
        ttk.Checkbutton(self.checkboxes_frame, text="yeti (ZX) HV", 
                       variable=self.checkbox_vars["yeti"], 
                       command=self.update_plot).pack(anchor="w")
        ttk.Checkbutton(self.checkboxes_frame, text="zuzia (XY) HV", 
                       variable=self.checkbox_vars["zuzia"], 
                       command=self.update_plot).pack(anchor="w")
        
        self.checkboxes_frame.update()
        print(f"Initial checkboxes created: {len(self.checkboxes_frame.winfo_children())} widgets")

    def _build_checkboxes_for_polarization(self):
        """Przebudowuje checkboxy w zależności od typu polaryzacji"""
        print(f"\nBuilding checkboxes for polarization type: {self.polarization_type}")
        
        for w in self.checkboxes_frame.winfo_children():
            w.destroy()
        
        if self.polarization_type == "circular":
            print("Creating CIRCULAR polarization checkboxes...")
            tk.Label(self.checkboxes_frame, text="Pokaż płaszczyzny (polaryzacja kołowa):", bg="#f5f5f5",
                     font=("Segoe UI",10,"bold")).pack(anchor="w", pady=(0,2))
            
            self.checkbox_vars = {
                "xawery_RHCP": tk.BooleanVar(value=True),
                "xawery_LHCP": tk.BooleanVar(value=True),
                "xawery_RL": tk.BooleanVar(value=False),
                "yeti_RHCP": tk.BooleanVar(value=True),
                "yeti_LHCP": tk.BooleanVar(value=True),
                "yeti_RL": tk.BooleanVar(value=False),
                "zuzia_RHCP": tk.BooleanVar(value=True),
                "zuzia_LHCP": tk.BooleanVar(value=True),
                "zuzia_RL": tk.BooleanVar(value=False),
            }
            
            # Xawery (YZ)
            tk.Label(self.checkboxes_frame, text="xawery (YZ):", bg="#f5f5f5", 
                     font=("Segoe UI",9,"bold")).pack(anchor="w", padx=(0,0), pady=(4,0))
            ttk.Checkbutton(self.checkboxes_frame, text="  RHCP", 
                           variable=self.checkbox_vars["xawery_RHCP"], 
                           command=self.update_plot).pack(anchor="w", padx=(10,0))
            ttk.Checkbutton(self.checkboxes_frame, text="  LHCP", 
                           variable=self.checkbox_vars["xawery_LHCP"], 
                           command=self.update_plot).pack(anchor="w", padx=(10,0))
            ttk.Checkbutton(self.checkboxes_frame, text="  R+L (suma mocy)", 
                           variable=self.checkbox_vars["xawery_RL"], 
                           command=self.update_plot).pack(anchor="w", padx=(10,0))
            
            # Yeti (ZX)
            tk.Label(self.checkboxes_frame, text="yeti (ZX):", bg="#f5f5f5", 
                     font=("Segoe UI",9,"bold")).pack(anchor="w", padx=(0,0), pady=(4,0))
            ttk.Checkbutton(self.checkboxes_frame, text="  RHCP", 
                           variable=self.checkbox_vars["yeti_RHCP"], 
                           command=self.update_plot).pack(anchor="w", padx=(10,0))
            ttk.Checkbutton(self.checkboxes_frame, text="  LHCP", 
                           variable=self.checkbox_vars["yeti_LHCP"], 
                           command=self.update_plot).pack(anchor="w", padx=(10,0))
            ttk.Checkbutton(self.checkboxes_frame, text="  R+L (suma mocy)", 
                           variable=self.checkbox_vars["yeti_RL"], 
                           command=self.update_plot).pack(anchor="w", padx=(10,0))
            
            # Zuzia (XY)
            tk.Label(self.checkboxes_frame, text="zuzia (XY):", bg="#f5f5f5", 
                     font=("Segoe UI",9,"bold")).pack(anchor="w", padx=(0,0), pady=(4,0))
            ttk.Checkbutton(self.checkboxes_frame, text="  RHCP", 
                           variable=self.checkbox_vars["zuzia_RHCP"], 
                           command=self.update_plot).pack(anchor="w", padx=(10,0))
            ttk.Checkbutton(self.checkboxes_frame, text="  LHCP", 
                           variable=self.checkbox_vars["zuzia_LHCP"], 
                           command=self.update_plot).pack(anchor="w", padx=(10,0))
            ttk.Checkbutton(self.checkboxes_frame, text="  R+L (suma mocy)", 
                           variable=self.checkbox_vars["zuzia_RL"], 
                           command=self.update_plot).pack(anchor="w", padx=(10,0))
            
            print(f"Created {len(self.checkbox_vars)} checkbox variables")
        else:
            print("Creating LINEAR (HV) polarization checkboxes...")
            # Linear (HV)
            tk.Label(self.checkboxes_frame, text="Pokaż płaszczyzny (HV):", bg="#f5f5f5",
                     font=("Segoe UI",10,"bold")).pack(anchor="w", pady=(0,2))
            
            self.checkbox_vars = {
                "xawery": tk.BooleanVar(value=True),
                "yeti": tk.BooleanVar(value=True),
                "zuzia": tk.BooleanVar(value=True)
            }
            
            ttk.Checkbutton(self.checkboxes_frame, text="xawery (YZ) HV", 
                           variable=self.checkbox_vars["xawery"], 
                           command=self.update_plot).pack(anchor="w")
            ttk.Checkbutton(self.checkboxes_frame, text="yeti (ZX) HV", 
                           variable=self.checkbox_vars["yeti"], 
                           command=self.update_plot).pack(anchor="w")
            ttk.Checkbutton(self.checkboxes_frame, text="zuzia (XY) HV", 
                           variable=self.checkbox_vars["zuzia"], 
                           command=self.update_plot).pack(anchor="w")
            
            print(f"Created {len(self.checkbox_vars)} checkbox variables")
        
        self.checkboxes_frame.update()
        print(f"Checkboxes frame now has {len(self.checkboxes_frame.winfo_children())} widgets")

    def update_legend(self, items):
        for w in self.legend_frame.winfo_children(): w.destroy()
        if not items:
            tk.Label(self.legend_frame, text="– brak –", bg="#f5f5f5", fg="#777").pack(anchor="w")
            return
        def add_row(color, text):
            row = tk.Frame(self.legend_frame, bg="#f5f5f5"); row.pack(anchor="w")
            sw = tk.Canvas(row, width=28, height=8, bg="#f5f5f5", highlightthickness=0)
            sw.pack(side=tk.LEFT, padx=(0,6), pady=2)
            sw.create_line(2,4,26,4, fill=color, width=3)
            tk.Label(row, text=text, bg="#f5f5f5").pack(side=tk.LEFT)
        for color, text in items:
            add_row(color, text)

    def pick_file_main(self):
        p = filedialog.askopenfilename(
            title="Wybierz plik MegiQ RMS",
            filetypes=[("Pliki tekstowe","*.txt;*.csv;*.dat"), ("Wszystkie","*.*")]
        )
        if p: self.path_var.set(p)

    def pick_file_pust(self):
        p = filedialog.askopenfilename(
            title="Wybierz plik Pustelnik (HV only)",
            filetypes=[("Pliki tekstowe","*.txt;*.csv;*.dat"), ("Wszystkie","*.*")]
        )
        if p: self.pust_path_var.set(p); self.var_pust.set(True)

    def load_files(self):
        print("\n" + "="*60)
        print("LOAD FILES CALLED")
        print("="*60)
        
        main_set = self.path_var.get().strip() != ""
        pust_set = self.pust_path_var.get().strip() != ""
        
        print(f"Main file set: {main_set}")
        print(f"Pustelnik file set: {pust_set}")
        
        if not main_set and not pust_set:
            self.status_label.config(text="Status: Błąd - brak wybranych plików!", fg="red")
            messagebox.showwarning("Brak plików", "Wskaż przynajmniej jeden plik: RMS lub Pustelnik.")
            return
            
        if main_set:
            file_path = self.path_var.get().strip()
            print(f"\nAttempting to load main file: {file_path}")
            self.status_label.config(text=f"Status: Wczytuję plik główny...", fg="blue")
            self.update()
            
            try: 
                self.data_main = parse_megiq_txt(file_path)
                self.polarization_type = self.data_main.get("polarization", "linear")
                
                print(f"\n✓ SUCCESS: File loaded")
                print(f"  Polarization type: {self.polarization_type}")
                print(f"  Angle points: {len(self.data_main['angle_deg'])}")
                
                self.status_label.config(
                    text=f"Status: ✓ Wczytano plik ({self.polarization_type}, {len(self.data_main['angle_deg'])} punktów)", 
                    fg="green"
                )
                self._build_checkboxes_for_polarization()
                
            except Exception as e:
                print(f"\n✗ ERROR loading main file:")
                print(f"  Exception type: {type(e).__name__}")
                print(f"  Exception message: {str(e)}")
                import traceback
                print(f"  Full traceback:")
                traceback.print_exc()
                
                self.status_label.config(text=f"Status: ✗ Błąd: {str(e)[:100]}", fg="red")
                messagebox.showerror("Błąd pliku RMS", 
                    f"Nie udało się odczytać pliku RMS:\n{file_path}\n\nBłąd: {str(e)}\n\nSprawdź konsolę aby zobaczyć szczegóły.")
                self.data_main = None
                self.polarization_type = "linear"
                self._build_checkboxes_for_polarization()
                return
                
        if pust_set:
            pust_path = self.pust_path_var.get().strip()
            print(f"\nAttempting to load pustelnik file: {pust_path}")
            self.status_label.config(text=f"Status: Wczytuję pustelnika...", fg="blue")
            self.update()
            
            try: 
                self.data_pust = parse_pustelnik_txt(pust_path)
                print(f"✓ Pustelnik loaded: {len(self.data_pust['angle_deg'])} points")
                
            except Exception as e:
                print(f"\n✗ ERROR loading pustelnik:")
                print(f"  Exception: {str(e)}")
                import traceback
                traceback.print_exc()
                
                messagebox.showerror("Błąd pustelnika", 
                    f"Nie udało się odczytać pustelnika:\n{pust_path}\n\n{e}")
                self.data_pust = None
        
        print("\nCalling update_plot()...")
        self.update_plot()

    def update_plot(self):
        print("\n" + "="*60)
        print("UPDATE PLOT CALLED")
        print("="*60)
        
        series_list, legend_items = [], []

        if self.data_main is not None:
            print(f"Main data available: polarization={self.polarization_type}")
            angle = self.data_main["angle_deg"]
            print(f"  Angle data points: {len(angle)}")
            
            if self.polarization_type == "circular":
                print("  Processing CIRCULAR polarization...")
                # Polaryzacja kołowa
                planes = [
                    ("xawery", "YZ", ["RHCP", "LHCP", "RL"]),
                    ("yeti", "ZX", ["RHCP", "LHCP", "RL"]),
                    ("zuzia", "XY", ["RHCP", "LHCP", "RL"])
                ]
                
                for plane_name, plane_key, pols in planes:
                    for pol in pols:
                        var_key = f"{plane_name}_{pol}"
                        is_checked = var_key in self.checkbox_vars and self.checkbox_vars[var_key].get()
                        print(f"    {var_key}: checked={is_checked}")
                        
                        if is_checked:
                            data = self.data_main[plane_key].get(pol)
                            if data is not None:
                                print(f"      → Adding to plot (data points: {len(data)})")
                                label_text = f"{plane_name} {pol}" if pol != "RL" else f"{plane_name} R+L"
                                series_list.append({
                                    "label": label_text,
                                    "angle_deg": angle,
                                    "val": data,
                                    "color": CIRCULAR_COLOR[var_key]
                                })
                                legend_items.append((CIRCULAR_COLOR[var_key], label_text))
                            else:
                                print(f"      ✗ Data is None")
            else:
                print("  Processing LINEAR polarization (HV)...")
                # Polaryzacja liniowa (HV)
                if "xawery" in self.checkbox_vars and self.checkbox_vars["xawery"].get():
                    if self.data_main["YZ"]["HV"] is not None:
                        print(f"    → Adding xawery HV")
                        series_list.append({
                            "label":"xawery HV",
                            "angle_deg":angle,
                            "val":self.data_main["YZ"]["HV"],
                            "color":HV_COLOR["xawery"]
                        })
                        legend_items.append((HV_COLOR["xawery"], "xawery (YZ) HV"))
                
                if "yeti" in self.checkbox_vars and self.checkbox_vars["yeti"].get():
                    if self.data_main["ZX"]["HV"] is not None:
                        print(f"    → Adding yeti HV")
                        series_list.append({
                            "label":"yeti HV",
                            "angle_deg":angle,
                            "val":self.data_main["ZX"]["HV"],
                            "color":HV_COLOR["yeti"]
                        })
                        legend_items.append((HV_COLOR["yeti"], "yeti (ZX) HV"))
                
                if "zuzia" in self.checkbox_vars and self.checkbox_vars["zuzia"].get():
                    if self.data_main["XY"]["HV"] is not None:
                        print(f"    → Adding zuzia HV")
                        series_list.append({
                            "label":"zuzia HV",
                            "angle_deg":angle,
                            "val":self.data_main["XY"]["HV"],
                            "color":HV_COLOR["zuzia"]
                        })
                        legend_items.append((HV_COLOR["zuzia"], "zuzia (XY) HV"))
        else:
            print("Main data is None")

        if self.var_pust.get() and self.data_pust is not None:
            print("Adding pustelnik data")
            series_list.append({
                "label":"pustelnik",
                "angle_deg":self.data_pust["angle_deg"],
                "val":self.data_pust["HV"],
                "color":HV_COLOR["pustelnik"]
            })
            legend_items.append((HV_COLOR["pustelnik"], "pustelnik HV"))

        print(f"\nTotal series to plot: {len(series_list)}")
        
        if not series_list:
            print("No series to plot - clearing axes")
            self.ax.clear(); self.ax.set_axis_off(); self.canvas.draw_idle(); self.update_legend([]); return

        mn, mx, st = RMS_SCALE
        title = "Overlay (stała skala −10..+10 dB)"
        if self.polarization_type == "circular":
            title = "RHCP/LHCP/R+L overlay (stała skala −10..+10 dB)"
        
        print(f"Drawing polar plot: title='{title}'")
        draw_polar(self.ax, series_list, title, mn, mx, st)
        self.canvas.draw_idle()
        self.update_legend(legend_items)
        print("✓ Plot updated successfully")

# ===================== ZAKŁADKA 2 – Theta/Phi (auto) =====================
class TabThetaPhi(ttk.Frame):
    def __init__(self, master, ax, canvas):
        super().__init__(master)
        self.ax, self.canvas = ax, canvas

        # każdy wiersz: {"path": str, "theta": np.array, "db": np.array, "var": BooleanVar, "color": str, "used": str}
        self.items = []
        self.colors = ["#d81b60","#1e88e5","#43a047","#fdd835","#8e24aa",
                       "#e53935","#00acc1","#7cb342","#fb8c00","#3949ab"]

        left = tk.Frame(self, bg="#f5f5f5"); left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        tk.Label(left, text="Wiele plików (Theta/Phi/Abs(Dir.)[dBi]) – auto łączenie przekrojów",
                 bg="#f5f5f5", font=("Segoe UI", 11, "bold"), wraplength=330).pack(anchor="w")

        bar = tk.Frame(left, bg="#f5f5f5"); bar.pack(anchor="w", pady=(4,6), fill=tk.X)
        ttk.Button(bar, text="Dodaj pliki…", command=self.add_files).pack(side=tk.LEFT)
        ttk.Button(bar, text="Usuń zaznaczone", command=self.remove_checked).pack(side=tk.LEFT, padx=6)
        ttk.Button(bar, text="Usuń ukryte", command=self.remove_unchecked).pack(side=tk.LEFT)
        ttk.Button(bar, text="Wyczyść", command=self.clear_all).pack(side=tk.RIGHT)

        self.list_frame = tk.LabelFrame(left, text="Pliki / widoczność", bg="#f5f5f5")
        self.list_frame.pack(anchor="w", fill=tk.BOTH, expand=True, pady=(6,4))

        tk.Label(left, text="Legenda (widoczne):", bg="#f5f5f5",
                 font=("Segoe UI",10,"bold")).pack(anchor="w")
        self.legend_frame = tk.Frame(left, bg="#f5f5f5")
        self.legend_frame.pack(anchor="w", fill=tk.X, pady=(2,8))

        tk.Label(left, text="Zoom: CTRL+przeciągnij (box), kółko – promień, Home – reset",
                 bg="#f5f5f5", fg="#444").pack(anchor="w", pady=(2,0))

    def _rebuild_list(self):
        for w in self.list_frame.winfo_children(): w.destroy()
        if not self.items:
            tk.Label(self.list_frame, text="– brak plików –", bg="#f5f5f5", fg="#777").pack(anchor="w", padx=6, pady=6)
            return
        for idx, it in enumerate(self.items):
            row = tk.Frame(self.list_frame, bg="#f5f5f5"); row.pack(anchor="w", fill=tk.X, padx=6, pady=2)
            sw = tk.Canvas(row, width=28, height=10, bg="#f5f5f5", highlightthickness=0)
            sw.pack(side=tk.LEFT, padx=(0,6)); sw.create_line(2,5,26,5, fill=it["color"], width=3)
            cb = ttk.Checkbutton(row, text=os.path.basename(it["path"]),
                                 variable=it["var"], command=self.draw)
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            if it.get("used"):
                tk.Label(row, text=f"({it['used']})", bg="#f5f5f5", fg="#666").pack(side=tk.LEFT, padx=4)
            ttk.Button(row, text="Usuń", width=7,
                       command=lambda i=idx: self.remove_index(i)).pack(side=tk.RIGHT)

    def _legend(self, items):
        for w in self.legend_frame.winfo_children(): w.destroy()
        if not items:
            tk.Label(self.legend_frame, text="– brak –", bg="#f5f5f5", fg="#777").pack(anchor="w"); return
        for col, lab in items:
            r = tk.Frame(self.legend_frame, bg="#f5f5f5"); r.pack(anchor="w")
            sw = tk.Canvas(r, width=28, height=8, bg="#f5f5f5", highlightthickness=0)
            sw.pack(side=tk.LEFT, padx=(0,6), pady=2); sw.create_line(2,4,26,4, fill=col, width=3)
            tk.Label(r, text=lab, bg="#f5f5f5").pack(side=tk.LEFT)

    # --- akcje listy ---
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Wybierz pliki Theta/Phi",
            filetypes=[("Pliki tekstowe","*.txt;*.dat;*.csv"), ("Wszystkie","*.*")]
        )
        for p in paths:
            try:
                df = parse_theta_phi_file(p)
                theta, db, used = auto_extract_tp_series(df, tol=2.0)
            except Exception as e:
                messagebox.showerror("Błąd pliku", f"{os.path.basename(p)}\n{e}")
                continue
            color = self.colors[len(self.items) % len(self.colors)]
            var = tk.BooleanVar(value=True)
            var.trace_add("write", lambda *_: self.draw())
            self.items.append({"path": p, "theta": theta, "db": db,
                               "used": used, "var": var, "color": color})
        self._rebuild_list()
        self.draw()

    def remove_index(self, idx):
        if 0 <= idx < len(self.items):
            self.items.pop(idx)
            self._rebuild_list()
            self.draw()

    def remove_checked(self):
        self.items = [it for it in self.items if not it["var"].get()]
        self._rebuild_list(); self.draw()

    def remove_unchecked(self):
        self.items = [it for it in self.items if it["var"].get()]
        self._rebuild_list(); self.draw()

    def clear_all(self):
        self.items.clear(); self._rebuild_list(); self.draw()

    # --- rysowanie ---
    def draw(self):
        vis = [it for it in self.items if it["var"].get()]
        if not vis:
            self.ax.clear(); self.ax.set_axis_off(); self.canvas.draw_idle(); self._legend([]); return

        series = []
        for it in vis:
            theta_plot = (360.0 - it["theta"]) % 360.0   # <- odwrócenie, żeby nie było lustrzanego odbicia
            series.append({
                "label": os.path.basename(it["path"]),
                "angle_deg": theta_plot,
                "val": it["db"],
                "color": it["color"]
            })

        mn, mx, st = TP_SCALE
        draw_polar(self.ax, series, "Abs(Dir.) [dBi] – auto φ (stała skala −40..+20 dB)", mn, mx, st)
        self.canvas.draw_idle()
        self._legend([(s["color"], s["label"]) for s in series])

# ============================ ZAKŁADKA 3 – WSZYSTKO RAZEM ===============================
class TabAll(ttk.Frame):
    def __init__(self, master, ax, canvas):
        super().__init__(master)
        self.ax, self.canvas = ax, canvas
        
        # Lista wszystkich serii: {"type": "rms"|"tp", "label": str, "angle": array, "val": array, 
        #                           "var": BooleanVar, "color": str, "source_path": str}
        self.items = []
        self.colors = ["#e91e63","#2196f3","#43a047","#ffd400","#9c27b0",
                       "#f44336","#00bcd4","#8bc34a","#ff9800","#3f51b5"]
        self.color_idx = 0

        left = tk.Frame(self, bg="#f5f5f5"); left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        tk.Label(left, text="Porównanie wszystkiego na jednym wykresie",
                 bg="#f5f5f5", font=("Segoe UI", 11, "bold"), wraplength=330).pack(anchor="w")

        # Przyciski ładowania
        bar = tk.Frame(left, bg="#f5f5f5"); bar.pack(anchor="w", pady=(8,6), fill=tk.X)
        ttk.Button(bar, text="+ RMS/Circular", command=self.add_rms).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(bar, text="+ Theta/Phi", command=self.add_tp).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(bar, text="+ Pustelnik", command=self.add_pustelnik).pack(side=tk.LEFT)
        
        bar2 = tk.Frame(left, bg="#f5f5f5"); bar2.pack(anchor="w", pady=(0,6), fill=tk.X)
        ttk.Button(bar2, text="Usuń zaznaczone", command=self.remove_checked).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(bar2, text="Wyczyść wszystko", command=self.clear_all).pack(side=tk.LEFT)

        # Lista plików/serii
        self.list_frame = tk.LabelFrame(left, text="Załadowane serie", bg="#f5f5f5")
        self.list_frame.pack(anchor="w", fill=tk.BOTH, expand=True, pady=(6,4))

        tk.Label(left, text="Legenda (widoczne):", bg="#f5f5f5",
                 font=("Segoe UI",10,"bold")).pack(anchor="w")
        self.legend_frame = tk.Frame(left, bg="#f5f5f5")
        self.legend_frame.pack(anchor="w", fill=tk.X, pady=(2,8))

        tk.Label(left, text="Zoom: CTRL+przeciągnij (box), kółko – promień, Home – reset",
                 bg="#f5f5f5", fg="#444").pack(anchor="w", pady=(2,0))
        
        self._rebuild_list()

    def _get_next_color(self):
        color = self.colors[self.color_idx % len(self.colors)]
        self.color_idx += 1
        return color

    def add_rms(self):
        """Dodaje plik RMS (HV lub circular) - pozwala wybrać płaszczyzny"""
        path = filedialog.askopenfilename(
            title="Wybierz plik RMS (HV lub Circular)",
            filetypes=[("Pliki tekstowe","*.txt;*.csv;*.dat"), ("Wszystkie","*.*")]
        )
        if not path:
            return
        
        try:
            data = parse_megiq_txt(path)
            pol_type = data.get("polarization", "linear")
            angle = data["angle_deg"]
            
            # Wybór płaszczyzn do dodania
            dialog = tk.Toplevel(self)
            dialog.title("Wybierz serie do dodania")
            dialog.geometry("400x350")
            dialog.transient(self)
            dialog.grab_set()
            
            tk.Label(dialog, text=f"Plik: {os.path.basename(path)}\nTyp: {pol_type}",
                     font=("Segoe UI", 10, "bold")).pack(pady=10)
            
            selections = {}
            
            if pol_type == "circular":
                for plane_name, plane_key in [("xawery (YZ)", "YZ"), ("yeti (ZX)", "ZX"), ("zuzia (XY)", "XY")]:
                    frame = tk.LabelFrame(dialog, text=plane_name, padx=10, pady=5)
                    frame.pack(fill=tk.X, padx=10, pady=5)
                    
                    for pol in ["RHCP", "LHCP", "RL"]:
                        if data[plane_key].get(pol) is not None:
                            var = tk.BooleanVar(value=True)
                            selections[f"{plane_key}_{pol}"] = (var, data[plane_key][pol], f"{plane_name} {pol}")
                            ttk.Checkbutton(frame, text=pol, variable=var).pack(anchor="w")
            else:
                for plane_name, plane_key in [("xawery (YZ)", "YZ"), ("yeti (ZX)", "ZX"), ("zuzia (XY)", "XY")]:
                    if data[plane_key].get("HV") is not None:
                        var = tk.BooleanVar(value=True)
                        selections[f"{plane_key}_HV"] = (var, data[plane_key]["HV"], f"{plane_name} HV")
                        ttk.Checkbutton(dialog, text=f"{plane_name} HV", variable=var).pack(anchor="w", padx=20, pady=2)
            
            def ok_clicked():
                for key, (var, val_data, label) in selections.items():
                    if var.get():
                        item_var = tk.BooleanVar(value=True)
                        item_var.trace_add("write", lambda *_: self.draw())
                        self.items.append({
                            "type": "rms",
                            "label": f"{os.path.basename(path)} - {label}",
                            "angle": angle,
                            "val": val_data,
                            "var": item_var,
                            "color": self._get_next_color(),
                            "source_path": path
                        })
                dialog.destroy()
                self._rebuild_list()
                self.draw()
            
            ttk.Button(dialog, text="OK", command=ok_clicked).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można wczytać pliku RMS:\n{e}")

    def add_tp(self):
        """Dodaje pliki Theta/Phi"""
        paths = filedialog.askopenfilenames(
            title="Wybierz pliki Theta/Phi",
            filetypes=[("Pliki tekstowe","*.txt;*.dat;*.csv"), ("Wszystkie","*.*")]
        )
        for p in paths:
            try:
                df = parse_theta_phi_file(p)
                theta, db, used = auto_extract_tp_series(df, tol=2.0)
                theta_plot = (360.0 - theta) % 360.0
                
                var = tk.BooleanVar(value=True)
                var.trace_add("write", lambda *_: self.draw())
                self.items.append({
                    "type": "tp",
                    "label": f"{os.path.basename(p)} ({used})",
                    "angle": theta_plot,
                    "val": db,
                    "var": var,
                    "color": self._get_next_color(),
                    "source_path": p
                })
            except Exception as e:
                messagebox.showerror("Błąd pliku", f"{os.path.basename(p)}\n{e}")
                continue
        
        self._rebuild_list()
        self.draw()

    def add_pustelnik(self):
        """Dodaje plik Pustelnik (HV)"""
        path = filedialog.askopenfilename(
            title="Wybierz plik Pustelnik",
            filetypes=[("Pliki tekstowe","*.txt;*.csv;*.dat"), ("Wszystkie","*.*")]
        )
        if not path:
            return
        
        try:
            data = parse_pustelnik_txt(path)
            var = tk.BooleanVar(value=True)
            var.trace_add("write", lambda *_: self.draw())
            self.items.append({
                "type": "pustelnik",
                "label": f"{os.path.basename(path)} (Pustelnik HV)",
                "angle": data["angle_deg"],
                "val": data["HV"],
                "var": var,
                "color": self._get_next_color(),
                "source_path": path
            })
            self._rebuild_list()
            self.draw()
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można wczytać pustelnika:\n{e}")

    def _rebuild_list(self):
        for w in self.list_frame.winfo_children(): 
            w.destroy()
        
        if not self.items:
            tk.Label(self.list_frame, text="– brak załadowanych danych –", 
                    bg="#f5f5f5", fg="#777").pack(anchor="w", padx=6, pady=6)
            return
        
        for idx, it in enumerate(self.items):
            row = tk.Frame(self.list_frame, bg="#f5f5f5")
            row.pack(anchor="w", fill=tk.X, padx=6, pady=2)
            
            # Color swatch
            sw = tk.Canvas(row, width=28, height=10, bg="#f5f5f5", highlightthickness=0)
            sw.pack(side=tk.LEFT, padx=(0,6))
            sw.create_line(2,5,26,5, fill=it["color"], width=3)
            
            # Checkbox
            cb = ttk.Checkbutton(row, text=it["label"], variable=it["var"])
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Remove button
            ttk.Button(row, text="Usuń", width=7,
                      command=lambda i=idx: self.remove_index(i)).pack(side=tk.RIGHT)

    def _legend(self, items):
        for w in self.legend_frame.winfo_children(): 
            w.destroy()
        
        if not items:
            tk.Label(self.legend_frame, text="– brak –", 
                    bg="#f5f5f5", fg="#777").pack(anchor="w")
            return
        
        for col, lab in items:
            r = tk.Frame(self.legend_frame, bg="#f5f5f5")
            r.pack(anchor="w")
            sw = tk.Canvas(r, width=28, height=8, bg="#f5f5f5", highlightthickness=0)
            sw.pack(side=tk.LEFT, padx=(0,6), pady=2)
            sw.create_line(2,4,26,4, fill=col, width=3)
            tk.Label(r, text=lab, bg="#f5f5f5").pack(side=tk.LEFT)

    def remove_index(self, idx):
        if 0 <= idx < len(self.items):
            self.items.pop(idx)
            self._rebuild_list()
            self.draw()

    def remove_checked(self):
        self.items = [it for it in self.items if not it["var"].get()]
        self._rebuild_list()
        self.draw()

    def clear_all(self):
        self.items.clear()
        self.color_idx = 0
        self._rebuild_list()
        self.draw()

    def draw(self):
        vis = [it for it in self.items if it["var"].get()]
        
        if not vis:
            self.ax.clear()
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            self._legend([])
            return

        series = []
        for it in vis:
            series.append({
                "label": it["label"],
                "angle_deg": it["angle"],
                "val": it["val"],
                "color": it["color"]
            })

        # Use wider scale to accommodate all types
        draw_polar(self.ax, series, "Porównanie wszystkich pomiarów", -20.0, 15.0, 5.0)
        self.canvas.draw_idle()
        self._legend([(s["color"], s["label"]) for s in series])

# ============================ APLIKACJA GŁÓWNA ===============================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nakładany wykres kołowy – RMS/Circular/Theta/Phi + Porównanie")
        self.geometry("1280x880")
        self.configure(bg="#f5f5f5")

        # prawa strona – wykres
        main = tk.Frame(self, bg="#f5f5f5"); main.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.fig = Figure(figsize=(8.8, 7.8), dpi=100, facecolor="white", constrained_layout=True)
        self.ax = self.fig.add_subplot(111, projection="polar")
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, main, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.boxzoom = BoxZoom(self.ax, self.canvas)

        # lewa kolumna – zakładki
        left_col = tk.Frame(self, bg="#f5f5f5"); left_col.pack(side=tk.LEFT, fill=tk.BOTH, padx=6, pady=6)

        self.nb = ttk.Notebook(left_col)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_rms = TabRMS(self.nb, self.ax, self.canvas)
        self.tab_tp  = TabThetaPhi(self.nb, self.ax, self.canvas)
        self.tab_all = TabAll(self.nb, self.ax, self.canvas)
        self.nb.add(self.tab_rms, text="RMS (HV/Circular) + Pustelnik")
        self.nb.add(self.tab_tp,  text="Wiele plików (Theta/Phi)")
        self.nb.add(self.tab_all, text="Porównanie wszystkiego")

    # scroll zoom (promień wokół kursora)
    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.ydata is None:
            return
        factor = 1.0/1.2 if event.step > 0 else 1.2
        rmin, rmax = self.ax.get_rmin(), self.ax.get_rmax()
        r0 = event.ydata
        new_rmin = r0 + (rmin - r0) * factor
        new_rmax = r0 + (rmax - r0) * factor
        if new_rmax - new_rmin < 1e-3:
            return
        self.ax.set_rlim(new_rmin, new_rmax)
        self.canvas.draw_idle()

if __name__ == "__main__":
    App().mainloop()
