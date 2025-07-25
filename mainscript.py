import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as mpatches
import skrf as rf
import pandas as pd
import tempfile
import numpy as np
import re
from itertools import combinations, cycle

MAXF = 4
MINF = 0.4

def loadFile(p):
    with open(p, "r", encoding="utf-8") as f:
        l = f.readlines()
    fx = []
    for ln in l:
        if ln.lstrip().startswith(("!", "#")):
            fx.append(ln)
        else:
            fx.append(ln.replace(",", "."))
    t = tempfile.NamedTemporaryFile("w+", delete=False, suffix=Path(p).suffix, encoding="utf-8")
    t.writelines(fx)
    t.flush()
    t.close()
    return rf.Network(t.name)

class ValidatedDoubleVar(tk.DoubleVar):
    """DoubleVar with validation and fallback to previous valid value"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_valid = self.get()
    
    def set(self, value):
        try:
            if isinstance(value, str):
                value = value.replace(',', '.')
            float_val = float(value)
            super().set(float_val)
            self._last_valid = float_val
        except (ValueError, TypeError):
            super().set(self._last_valid)
            return False
        return True

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("i can see through your skin")
        self.geometry("1400x800")
        self.fls = []
        self.mrk = []
        self.mtxt = []
        self.fmin = ValidatedDoubleVar(value=MINF)
        self.fmax = ValidatedDoubleVar(value=MAXF)
        self.s11 = tk.BooleanVar(value=False)
        self.s22 = tk.BooleanVar(value=False)
        self.s12 = tk.BooleanVar(value=False)
        self.s21 = tk.BooleanVar(value=False)
        self.td = tk.StringVar(value="s11")
        self.cbox = None
        self.rbox = None
        self.fax = []
        self.tax = []
        self.tab = "freq"
        self.gateChk = tk.BooleanVar(value=False)
        self.gateCenter = ValidatedDoubleVar(value=5.0)
        self.gateSpan = ValidatedDoubleVar(value=0.5)
        
        self.regexPattern = tk.StringVar(value=r'_(\d+)ml')
        self.regexParam = tk.StringVar(value="s21")
        self.regexHighlightChk = tk.BooleanVar(value=True)
        self.regexStrictMonotonic = tk.BooleanVar(value=False)  # Strict monotonicity
        self.regexGateChk = tk.BooleanVar(value=False)
        self.regexGroup = tk.IntVar(value=1)
        self.regexSpans = []
        self.regexLines = []
        self.regexRanges = []
        
        self.overlapData = {}
        self.fileListCanvas = None
        
        self.varS11 = tk.BooleanVar(value=True)
        self.varS22 = tk.BooleanVar(value=False)
        self.varS12 = tk.BooleanVar(value=False)
        self.varS21 = tk.BooleanVar(value=True)
        self.varMag = tk.BooleanVar(value=True)
        self.varPhase = tk.BooleanVar(value=True)
        self.varDetrend = tk.BooleanVar(value=True)  # Default to detrending
        self.varData = None
        self.varMrk = []
        self.varMtxt = []
        self._blkNextVar = False
        
        self._makeUi()
        self._updAll()

    def _validateNumeric(self, widget, var, callback=None):
        """Add validation to numeric entry widget"""
        def validate_and_update(*args):
            val = widget.get()
            if var.set(val):
                if callback:
                    callback()
            else:
                widget.delete(0, tk.END)
                widget.insert(0, str(var.get()))
                
        widget.bind("<FocusOut>", validate_and_update)
        widget.bind("<Return>", validate_and_update)

    def _makeUi(self):
        lfrm = ttk.Frame(self)
        lfrm.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        ffrm = ttk.Frame(lfrm)
        ffrm.pack(anchor="w", pady=(10,0))
        ttk.Label(ffrm, text="Zakres f [GHz]:").pack(side=tk.LEFT)
        ttk.Label(ffrm, text="Od").pack(side=tk.LEFT, padx=(5,0))
        ent1 = ttk.Entry(ffrm, textvariable=self.fmin, width=6)
        ent1.pack(side=tk.LEFT, padx=(2,10))
        self._validateNumeric(ent1, self.fmin, self._updAll)
        ttk.Label(ffrm, text="Do").pack(side=tk.LEFT)
        ent2 = ttk.Entry(ffrm, textvariable=self.fmax, width=6)
        ent2.pack(side=tk.LEFT, padx=(2,10))
        self._validateNumeric(ent2, self.fmax, self._updAll)
        btn1 = ttk.Button(lfrm, text="Dodaj pliki", command=self._addFiles)
        btn1.pack(anchor="w", pady=(0,10))
        btn2 = ttk.Button(lfrm, text="Usuń wszystkie markery", command=self._clearM)
        btn2.pack(anchor="w", pady=(0,10))
        btn3 = ttk.Button(lfrm, text="Usuń zaznaczone pliki", command=self._deleteSelectedFiles)
        btn3.pack(anchor="w", pady=(0,10))
        
        fileListFrame = ttk.Frame(lfrm)
        fileListFrame.pack(anchor="w", fill=tk.BOTH, pady=(0,10))
        
        fileListScroll = ttk.Scrollbar(fileListFrame)
        fileListScroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        fileListCanvas = tk.Canvas(fileListFrame, yscrollcommand=fileListScroll.set, width=250, height=200)
        fileListCanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        fileListScroll.config(command=fileListCanvas.yview)
        
        self.fbox = ttk.Frame(fileListCanvas)
        fileListCanvas.create_window((0,0), window=self.fbox, anchor="nw")
        
        self.fileListCanvas = fileListCanvas
        
        def updateScrollRegion(event=None):
            fileListCanvas.configure(scrollregion=fileListCanvas.bbox("all"))
        
        self.fbox.bind("<Configure>", updateScrollRegion)
        
        def onMouseWheel(event):
            fileListCanvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def onMouseWheelLinux(event):
            if event.num == 4:
                fileListCanvas.yview_scroll(-1, "units")
            elif event.num == 5:
                fileListCanvas.yview_scroll(1, "units")
        
        fileListCanvas.bind("<MouseWheel>", onMouseWheel)
        fileListCanvas.bind("<Button-4>", onMouseWheelLinux)
        fileListCanvas.bind("<Button-5>", onMouseWheelLinux)
        self.txt = tk.Text(lfrm, height=12, width=45, font=("Consolas", 9))
        self.txt.pack(anchor="w", pady=(20, 0))
        self.txt.config(state=tk.DISABLED)
        self.cbox = ttk.Frame(lfrm)
        ttk.Label(self.cbox, text="Pokaż na wykresie:").pack(side=tk.LEFT)
        for n, v in (("s11", self.s11), ("s22", self.s22), ("s12", self.s12), ("s21", self.s21)):
            chk = ttk.Checkbutton(self.cbox, text=n.upper(), variable=v, command=self._updAll)
            chk.pack(side=tk.LEFT, padx=2)
        self.rbox = ttk.Frame(lfrm)
        ttk.Label(self.rbox, text="Wybierz TDG:").pack(side=tk.LEFT)
        for n in ("s11", "s12", "s21", "s22"):
            rb = ttk.Radiobutton(self.rbox, text=n.upper(), value=n, variable=self.td, command=self._updAll)
            rb.pack(side=tk.LEFT, padx=2)
        self.gateBox = ttk.Frame(lfrm)
        self.gateCheck = ttk.Checkbutton(self.gateBox, text="Gating", variable=self.gateChk, command=self._updAll)
        self.gateCheck.pack(side=tk.LEFT, padx=3)
        ttk.Label(self.gateBox, text="Center [ns]").pack(side=tk.LEFT, padx=2)
        self.gateCenterEntry = ttk.Entry(self.gateBox, textvariable=self.gateCenter, width=5)
        self.gateCenterEntry.pack(side=tk.LEFT, padx=3)
        self._validateNumeric(self.gateCenterEntry, self.gateCenter, self._updAll)
        ttk.Label(self.gateBox, text="Span [ns]").pack(side=tk.LEFT, padx=2)
        self.gateSpanEntry = ttk.Entry(self.gateBox, textvariable=self.gateSpan, width=5)
        self.gateSpanEntry.pack(side=tk.LEFT, padx=3)
        self._validateNumeric(self.gateSpanEntry, self.gateSpan, self._updAll)
        self.cbox.pack(anchor="w", pady=(2,0))
        self.openGatedBtn = ttk.Button(self.gateBox, text="Otwórz gated (freq) w Matplotlib", command=self._showGatedPlot)
        self.openGatedBtn.pack(side=tk.LEFT, padx=8)
        
        self.regexBox = ttk.Frame(lfrm)
        
        ttk.Label(self.regexBox, text="Regex pattern:").pack(anchor="w")
        self.regexEntry = ttk.Entry(self.regexBox, textvariable=self.regexPattern, width=40)
        self.regexEntry.pack(anchor="w", pady=(2,5))
        self.regexEntry.bind("<Return>", lambda e: self._updRegexPlot())
        self.regexEntry.bind("<KeyRelease>", self._onRegexChange)
        
        helpFrame = ttk.Frame(self.regexBox)
        helpFrame.pack(anchor="w", pady=(0,5))
        helpText = ttk.Label(helpFrame, text="Pattern must have capture group (...) with number", 
                            font=("", 8), foreground="gray")
        helpText.pack(anchor="w")
        
        groupFrame = ttk.Frame(self.regexBox)
        groupFrame.pack(anchor="w", pady=(0,5))
        ttk.Label(groupFrame, text="Capture group:").pack(side=tk.LEFT)
        self.regexGroup = tk.IntVar(value=1)
        self.groupSpin = ttk.Spinbox(groupFrame, from_=1, to=10, width=5, 
                                     textvariable=self.regexGroup, command=self._updRegexPlot)
        self.groupSpin.pack(side=tk.LEFT, padx=(5,0))
        ttk.Label(groupFrame, text="(which group has the number)", font=("", 8)).pack(side=tk.LEFT, padx=(5,0))
        
        sparamFrame = ttk.Frame(self.regexBox)
        sparamFrame.pack(anchor="w", pady=(5,0))
        ttk.Label(sparamFrame, text="S-param:").pack(side=tk.LEFT)
        for n in ("s11", "s12", "s21", "s22"):
            rb = ttk.Radiobutton(sparamFrame, text=n.upper(), value=n, variable=self.regexParam, command=self._updRegexPlot)
            rb.pack(side=tk.LEFT, padx=2)
        
        regexCtrlFrame = ttk.Frame(self.regexBox)
        regexCtrlFrame.pack(anchor="w", pady=(5,0))
        self.regexHighlightCheckBtn = ttk.Checkbutton(regexCtrlFrame, text="Highlight monotonic", variable=self.regexHighlightChk, command=self._updRegexPlot)
        self.regexHighlightCheckBtn.pack(side=tk.LEFT, padx=3)
        self.regexStrictCheckBtn = ttk.Checkbutton(regexCtrlFrame, text="Strict monotonic", variable=self.regexStrictMonotonic, command=self._updRegexPlot)
        self.regexStrictCheckBtn.pack(side=tk.LEFT, padx=3)
        self.exportRangesBtn = ttk.Button(regexCtrlFrame, text="Export ranges", command=self._exportRanges)
        self.exportRangesBtn.pack(side=tk.LEFT, padx=3)
        
        # Add help text for strict monotonic
        strictHelpFrame = ttk.Frame(self.regexBox)
        strictHelpFrame.pack(anchor="w", pady=(2,0))
        ttk.Label(strictHelpFrame, text="Strict: no equal consecutive values; Non-strict: allows equal values", 
                 font=("", 8), foreground="gray").pack(anchor="w")
        
        # Gating controls for regex
        gatingFrame = ttk.Frame(self.regexBox)
        gatingFrame.pack(anchor="w", pady=(3,0))
        self.regexGateChk = tk.BooleanVar(value=False)
        self.regexGateCheck = ttk.Checkbutton(gatingFrame, text="Apply gating", variable=self.regexGateChk, command=self._updRegexPlot)
        self.regexGateCheck.pack(side=tk.LEFT, padx=3)
        ttk.Label(gatingFrame, text="Center [ns]").pack(side=tk.LEFT, padx=(10,2))
        self.regexGateCenterEntry = ttk.Entry(gatingFrame, textvariable=self.gateCenter, width=5)
        self.regexGateCenterEntry.pack(side=tk.LEFT, padx=3)
        self._validateNumeric(self.regexGateCenterEntry, self.gateCenter, self._updRegexPlot)
        ttk.Label(gatingFrame, text="Span [ns]").pack(side=tk.LEFT, padx=(5,2))
        self.regexGateSpanEntry = ttk.Entry(gatingFrame, textvariable=self.gateSpan, width=5)
        self.regexGateSpanEntry.pack(side=tk.LEFT, padx=3)
        self._validateNumeric(self.regexGateSpanEntry, self.gateSpan, self._updRegexPlot)
        
        patternFrame = ttk.Frame(self.regexBox)
        patternFrame.pack(anchor="w", pady=(3,0))
        ttk.Label(patternFrame, text="Quick patterns:").pack(anchor="w")
        
        patterns1 = ttk.Frame(patternFrame)
        patterns1.pack(anchor="w", pady=(2,0))
        ttk.Button(patterns1, text="_XXml", command=lambda: self._setRegex(r'_(\d+)ml')).pack(side=tk.LEFT, padx=2)
        ttk.Button(patterns1, text="XXmm", command=lambda: self._setRegex(r'(\d+)mm')).pack(side=tk.LEFT, padx=2)
        ttk.Button(patterns1, text="XXcm", command=lambda: self._setRegex(r'(\d+)cm')).pack(side=tk.LEFT, padx=2)
        
        patterns2 = ttk.Frame(patternFrame)
        patterns2.pack(anchor="w", pady=(2,0))
        ttk.Button(patterns2, text="_XX_", command=lambda: self._setRegex(r'_(\d+)_')).pack(side=tk.LEFT, padx=2)
        ttk.Button(patterns2, text="sample_XX", command=lambda: self._setRegex(r'sample_(\d+)')).pack(side=tk.LEFT, padx=2)
        ttk.Button(patterns2, text="XX.YYml", command=lambda: self._setRegex(r'(\d+\.?\d*)ml')).pack(side=tk.LEFT, padx=2)
        
        advFrame = ttk.Frame(self.regexBox)
        advFrame.pack(anchor="w", pady=(3,0))
        ttk.Label(advFrame, text="Advanced examples:", font=("", 9, "bold")).pack(anchor="w")
        exampleText = tk.Text(advFrame, height=4, width=40, font=("Consolas", 8))
        exampleText.pack(anchor="w", pady=(2,0))
        exampleText.insert(tk.END, "dist_(\d+)mm_vol_(\d+)ml  # group 1 or 2\n")
        exampleText.insert(tk.END, "test_v(\d+)_d(\d+)       # group 1 or 2\n")
        exampleText.insert(tk.END, "(\d+)_(\w+)_(\d+)ml      # group 1 or 3\n")
        exampleText.insert(tk.END, r"[^0-9]*(\d+)[^0-9]*$     # last number")
        exampleText.config(state=tk.DISABLED)
        
        self.overlapBox = ttk.Frame(lfrm)
        ttk.Label(self.overlapBox, text="Select frequency range files:").pack(anchor="w", pady=(5,0))
        btnFrame = ttk.Frame(self.overlapBox)
        btnFrame.pack(anchor="w", pady=(5,0))
        ttk.Button(btnFrame, text="Load range files", command=self._loadRangeFiles).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(btnFrame, text="Analyze from regex", command=self._analyzeFromRegex).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(btnFrame, text="Export overlaps", command=self._exportOverlaps).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(btnFrame, text="Clear plot", command=self._clearOverlapPlot).pack(side=tk.LEFT)
        
        self.varBox = ttk.Frame(lfrm)
        ttk.Label(self.varBox, text="S-parameters to analyze:").pack(anchor="w", pady=(5,0))
        sparamFrame = ttk.Frame(self.varBox)
        sparamFrame.pack(anchor="w", pady=(2,0))
        ttk.Checkbutton(sparamFrame, text="S11", variable=self.varS11, command=self._updVariancePlot).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(sparamFrame, text="S12", variable=self.varS12, command=self._updVariancePlot).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(sparamFrame, text="S21", variable=self.varS21, command=self._updVariancePlot).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(sparamFrame, text="S22", variable=self.varS22, command=self._updVariancePlot).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(self.varBox, text="Component type:").pack(anchor="w", pady=(5,0))
        compFrame = ttk.Frame(self.varBox)
        compFrame.pack(anchor="w", pady=(2,0))
        ttk.Checkbutton(compFrame, text="Magnitude", variable=self.varMag, command=self._updVariancePlot).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(compFrame, text="Phase", variable=self.varPhase, command=self._updVariancePlot).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(self.varBox, text="Phase processing:").pack(anchor="w", pady=(5,0))
        phaseFrame = ttk.Frame(self.varBox)
        phaseFrame.pack(anchor="w", pady=(2,0))
        ttk.Checkbutton(phaseFrame, text="Remove linear trend", variable=self.varDetrend, command=self._updVariancePlot).pack(side=tk.LEFT, padx=2)
        ttk.Label(phaseFrame, text="(removes cable delay effects)", font=("", 8), foreground="gray").pack(side=tk.LEFT, padx=(5,0))
        
        btnFrame2 = ttk.Frame(self.varBox)
        btnFrame2.pack(anchor="w", pady=(5,0))
        ttk.Button(btnFrame2, text="Clear markers", command=self._clearVarMarkers).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(btnFrame2, text="Export variance data", command=self._exportVariance).pack(side=tk.LEFT)

        nb = ttk.Notebook(self)
        nb.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.nb = nb
        frmF = ttk.Frame(nb)
        nb.add(frmF, text="Wykresy")
        self.nb.bind("<<NotebookTabChanged>>", self._onTab)
        self.fig, self.ax = plt.subplots(figsize=(10,7))
        self.cv = FigureCanvasTkAgg(self.fig, master=frmF)
        self.cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tb = NavigationToolbar2Tk(self.cv, frmF)
        self.tb.update()
        self.tb.pack(fill=tk.X)
        
        frmT = ttk.Frame(nb)
        nb.add(frmT, text="Time domain")
        self.figT, self.axT = plt.subplots(figsize=(10,10))
        self.cvT = FigureCanvasTkAgg(self.figT, master=frmT)
        self.cvT.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tbT = NavigationToolbar2Tk(self.cvT, frmT)
        self.tbT.update()
        self.tbT.pack(fill=tk.X)
        
        frmR = ttk.Frame(nb)
        nb.add(frmR, text="Regex Highlighting")
        self.figR, self.axR = plt.subplots(figsize=(10,7))
        self.cvR = FigureCanvasTkAgg(self.figR, master=frmR)
        self.cvR.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tbR = NavigationToolbar2Tk(self.cvR, frmR)
        self.tbR.update()
        self.tbR.pack(fill=tk.X)
        
        frmO = ttk.Frame(nb)
        nb.add(frmO, text="Range Overlaps")
        self.figO, self.axO = plt.subplots(figsize=(10,7))
        self.cvO = FigureCanvasTkAgg(self.figO, master=frmO)
        self.cvO.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tbO = NavigationToolbar2Tk(self.cvO, frmO)
        self.tbO.update()
        self.tbO.pack(fill=tk.X)
        
        frmV = ttk.Frame(nb)
        nb.add(frmV, text="Variance Analysis")
        # Increase figure size and add more padding
        self.figV, self.axV = plt.subplots(figsize=(10,8))
        self.cvV = FigureCanvasTkAgg(self.figV, master=frmV)
        self.cvV.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tbV = NavigationToolbar2Tk(self.cvV, frmV)
        self.tbV.update()
        self.tbV.pack(fill=tk.X)
        
        self.cv.mpl_connect('button_press_event', self._onClick)
        self.cvT.mpl_connect('button_press_event', self._onClick)
        self.cv.mpl_connect('pick_event', self._onPick)
        self.cvT.mpl_connect('pick_event', self._onPick)
        
        self.cvV.mpl_connect('button_press_event', self._onClickVar)
        self.cvV.mpl_connect('pick_event', self._onPickVar)
        self.cvV.mpl_connect('motion_notify_event', self._onMotionVar)

    def _setRegex(self, pattern):
        self.regexPattern.set(pattern)
        self._updRegexPlot()

    def _extractValue(self, filename, pattern):
        try:
            match = re.search(pattern, filename)
            if match:
                group_idx = self.regexGroup.get()
                if group_idx <= len(match.groups()):
                    value_str = match.group(group_idx)
                    return float(value_str)
        except:
            pass
        return None

    def _onRegexChange(self, event=None):
        pattern = self.regexPattern.get()
        try:
            re.compile(pattern)
            self.regexEntry.configure(foreground="black")
        except:
            self.regexEntry.configure(foreground="red")

    def _analyzeOrderedRanges(self, files_data):
        if not files_data:
            return [], None, None, []
            
        files_data.sort(key=lambda x: x[1])
        labels = [d[0] for d in files_data]
        freqs = files_data[0][2]
        s_matrix = np.array([d[3] for d in files_data])
        
        ordered_indices = []
        strict_monotonic = self.regexStrictMonotonic.get()
        
        for i in range(s_matrix.shape[1]):
            values = s_matrix[:, i]
            diffs = np.diff(values)
            
            if strict_monotonic:
                # Strict monotonicity: all differences must be strictly positive or negative
                if np.all(diffs > 0) or np.all(diffs < 0):
                    ordered_indices.append(i)
            else:
                # Non-strict monotonicity: allow equal values
                if np.all(diffs >= 0) or np.all(diffs <= 0):
                    ordered_indices.append(i)
        
        freq_ranges = []
        if ordered_indices:
            start = ordered_indices[0]
            for i in range(1, len(ordered_indices)):
                if ordered_indices[i] != ordered_indices[i - 1] + 1:
                    end = ordered_indices[i - 1]
                    freq_ranges.append((freqs[start], freqs[end]))
                    start = ordered_indices[i]
            freq_ranges.append((freqs[start], freqs[ordered_indices[-1]]))
        
        return freq_ranges, freqs, s_matrix, labels

    def _updRegexPlot(self):
        self.figR.clear()
        
        pattern = self.regexPattern.get()
        param = self.regexParam.get()
        
        self.regexSpans = []
        
        files_data = []
        all_files_info = []  
        
        for v, p, d in self.fls:
            fname = Path(p).stem
            value = self._extractValue(fname, pattern)
            
            if v.get():
                if value is not None:
                    all_files_info.append(f"✓ {fname} → {value}")
                else:
                    all_files_info.append(f"✗ {fname} → no match")
            
            if not v.get() or value is None:
                continue
                
            ext = Path(p).suffix.lower()
            
            try:
                if ext in ['.s1p', '.s2p', '.s3p']:
                    ntw = loadFile(p)
                    
                    if self.regexGateChk.get():
                        center = self.gateCenter.get()
                        span = self.gateSpan.get()
                        s_param = getattr(ntw, param)
                        s_gated = s_param.time_gate(center=center, span=span)
                        freq = ntw.f
                        s_db = s_gated.s_db.flatten()
                    else:
                        freq = ntw.f
                        s_db = getattr(ntw, param).s_db.flatten()
                    
                    files_data.append((fname, value, freq, s_db))
            except:
                pass
        
        self.txt.config(state=tk.NORMAL)
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, f"Regex: {pattern} (group {self.regexGroup.get()})\n")
        if self.regexGateChk.get():
            self.txt.insert(tk.END, f"Gating: {self.gateCenter.get()}±{self.gateSpan.get()/2} ns\n")
        if self.regexHighlightChk.get():
            mode = "Strict monotonic" if self.regexStrictMonotonic.get() else "Non-strict monotonic"
            self.txt.insert(tk.END, f"Highlighting: {mode}\n")
        self.txt.insert(tk.END, "-" * 40 + "\n")
        for info in all_files_info:
            self.txt.insert(tk.END, info + "\n")
        self.txt.config(state=tk.DISABLED)
        
        self.axR = self.figR.add_subplot(111)
        
        if not files_data:
            self.axR.text(0.5, 0.5, f"No files match pattern: {pattern}\nCheck regex and capture group", 
                         ha="center", va="center", fontsize=12, color="gray")
            self.axR.set_xticks([])
            self.axR.set_yticks([])
            self.cvR.draw()
            return
        
        ranges, freqs, s_matrix, labels = self._analyzeOrderedRanges(files_data)
        self.regexRanges = ranges  
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        
        self.regexLines = []
        for i, (s_data, label) in enumerate(zip(s_matrix, labels)):
            line, = self.axR.plot(freqs, s_data, label=label, color=colors[i])
            self.regexLines.append(line)
        
        if self.regexHighlightChk.get():
            for (f1, f2) in ranges:
                span = self.axR.axvspan(f1, f2, color='yellow', alpha=0.3)
                self.regexSpans.append(span)
        
        self.axR.set_xlabel("Frequency [Hz]")
        self.axR.set_ylabel(f"|{param.upper()}| [dB]")
        title = f"Regex-based ordering: {pattern} (group {self.regexGroup.get()})"
        if self.regexGateChk.get():
            title += f" - Gated ({self.gateCenter.get()}±{self.gateSpan.get()/2} ns)"
        self.axR.set_title(title)
        self.axR.grid(True)
        self.axR.legend()
        
        self.figR.tight_layout()
        self.cvR.draw()
        
    def _showGatedPlot(self):
        prm = self.td.get()
        center = self.gateCenter.get()
        span = self.gateSpan.get()
        plt.figure(figsize=(10,4))
        legTG = []
        for v, p, d in self.fls:
            if not v.get(): continue
            ext = Path(p).suffix.lower()
            lbl = Path(p).stem
            try:
                if ext in ['.s1p', '.s2p', '.s3p']:
                    ntw = loadFile(p)
                    sRaw = getattr(ntw, prm)
                    sGate = sRaw.time_gate(center=center, span=span)
                    freq = ntw.f
                    arr = sGate.s_db.flatten()
                    plt.plot(freq, arr, label=lbl)
                    legTG.append(lbl)
            except Exception:
                pass
        plt.ylabel(f"{prm.upper()} [dB]")
        plt.xlabel("Frequency [Hz]")
        plt.title(f"{prm.upper()} (frequency domain - gated)")
        plt.grid(True)
        if legTG:
            plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def _onTab(self, e):
        tabTxt = self.nb.tab(self.nb.select(), "text")
        if tabTxt == "Wykresy":
            self.tab = "freq"
            self.rbox.pack_forget()
            self.gateBox.pack_forget()
            self.regexBox.pack_forget()
            self.overlapBox.pack_forget()
            self.varBox.pack_forget()
            self.cbox.pack(anchor="w", pady=(2,0))
            self.txt.config(state=tk.NORMAL)
            self.txt.delete(1.0, tk.END)
            for txt in self.mtxt:
                self.txt.insert(tk.END, txt)
            self.txt.config(state=tk.DISABLED)
        elif tabTxt == "Time domain":
            self.tab = "time"
            self.cbox.pack_forget()
            self.regexBox.pack_forget()
            self.overlapBox.pack_forget()
            self.varBox.pack_forget()
            self.rbox.pack(anchor="w", pady=(2,0))
            self.gateBox.pack(anchor="w", pady=(8,0))
            self.txt.config(state=tk.NORMAL)
            self.txt.delete(1.0, tk.END)
            for txt in self.mtxt:
                self.txt.insert(tk.END, txt)
            self.txt.config(state=tk.DISABLED)
        elif tabTxt == "Regex Highlighting":
            self.tab = "regex"
            self.cbox.pack_forget()
            self.rbox.pack_forget()
            self.gateBox.pack_forget()
            self.overlapBox.pack_forget()
            self.varBox.pack_forget()
            self.regexBox.pack(anchor="w", pady=(2,0))
            self._updRegexPlot()
        elif tabTxt == "Range Overlaps":
            self.tab = "overlap"
            self.cbox.pack_forget()
            self.rbox.pack_forget()
            self.gateBox.pack_forget()
            self.regexBox.pack_forget()
            self.varBox.pack_forget()
            self.overlapBox.pack(anchor="w", pady=(2,0))
            self._updOverlapPlot()
        elif tabTxt == "Variance Analysis":
            self.tab = "variance"
            self.cbox.pack_forget()
            self.rbox.pack_forget()
            self.gateBox.pack_forget()
            self.regexBox.pack_forget()
            self.overlapBox.pack_forget()
            self.varBox.pack(anchor="w", pady=(2,0))
            if self.varData:
                mean_var = np.mean(self.varData['total_variance'])
                std_var = np.std(self.varData['total_variance'])
                self.txt.config(state=tk.NORMAL)
                self.txt.delete(1.0, tk.END)
                self.txt.insert(tk.END, f"Variance Statistics (Normalized Data):\n")
                self.txt.insert(tk.END, f"Normalization: {self.varData['normalization_info']}\n")
                self.txt.insert(tk.END, f"Mean total variance: {mean_var:.6f}\n")
                self.txt.insert(tk.END, f"Std of variance: {std_var:.6f}\n")
                self.txt.insert(tk.END, f"Files analyzed: {self.varData['file_count']}\n")
                self.txt.insert(tk.END, "-" * 40 + "\n")
                for txt in self.varMtxt:
                    self.txt.insert(tk.END, txt)
                self.txt.config(state=tk.DISABLED)
            self._updVariancePlot()

    def _addFiles(self):
        pth = filedialog.askopenfilenames(
            title="Wybierz pliki Touchstone (.sNp, .s1p, .csv)",
            filetypes=[
                ("Dane S-Param", "*.s1p *.s2p *.s3p *.csv"),
                ("Wszystkie pliki", "*.*")
            ]
        )
        for p in pth:
            if not any(p == f[1] for f in self.fls):
                v = tk.BooleanVar(value=True)
                chk = ttk.Checkbutton(self.fbox, text=Path(p).name, variable=v, command=self._updAll)
                chk.pack(anchor="w")
                self.fls.append((v, p, {}))
        self.fileListCanvas.configure(scrollregion=self.fileListCanvas.bbox("all"))
        self._updAll()

    def _updPlot(self):
        fmin = self.fmin.get()
        fmax = self.fmax.get()
        sstr = f"{fmin}-{fmax}ghz"
        sel = []
        if self.s11.get(): sel.append("s11")
        if self.s22.get(): sel.append("s22")
        if self.s12.get(): sel.append("s12")
        if self.s21.get(): sel.append("s21")
        self.fig.clf()
        if len(sel) == 0:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Zaznacz co najmniej jeden parametr\nżeby zobaczyć wykres", ha="center", va="center", fontsize=12, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            self.cv.draw()
            return
        axes = self.fig.subplots(len(sel), 1, sharex=True)
        self.fax = list(axes) if isinstance(axes, (list, np.ndarray)) else [axes]
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        for ax, prm in zip(axes, sel):
            leg = []
            for v, p, d in self.fls:
                if not v.get(): continue
                ext = Path(p).suffix.lower()
                lbl = Path(p).stem
                try:
                    if ext in ['.s1p', '.s2p', '.s3p']:
                        ntw = loadFile(p)[sstr]
                        d['ntwk'] = ntw
                        arr = getattr(ntw, prm).s_db.flatten()
                        fr = ntw.f
                    elif ext == '.csv' and prm == 's11':
                        df = pd.read_csv(p)
                        fr = df.iloc[:,0].values
                        arr = df.iloc[:,1].values
                        d['ntwk'] = None
                    else:
                        continue
                    ax.plot(fr, arr, label=lbl)
                    leg.append(lbl)
                except Exception:
                    pass
            ax.set_ylabel(f"|{prm.upper()}| [dB]")
            ax.set_title(f"{prm.upper()} magnitude")
            ax.grid(True)
            if leg:
                ax.legend(loc="upper right")
        axes[-1].set_xlabel("Frequency [Hz]")
        self.fig.tight_layout(rect=[0, 0, 1, 1])
        self.cv.draw()

    def _updTPlot(self):
        self.figT.clf()
        prm = self.td.get()
        center = self.gateCenter.get()
        span = self.gateSpan.get()
        doGate = self.gateChk.get()
        axes = self.figT.subplots(3, 1, sharex=False)
        self.tax = list(axes)
        axF = axes[0]
        axTR = axes[1]
        axTG = axes[2]
        legF = []
        legTR = []
        legTG = []
        # freq domain raw
        for v, p, d in self.fls:
            if not v.get(): continue
            ext = Path(p).suffix.lower()
            lbl = Path(p).stem
            try:
                if ext in ['.s1p', '.s2p', '.s3p']:
                    ntw = loadFile(p)
                    arr = getattr(ntw, prm).s_db.flatten()
                    fr = ntw.f
                    axF.plot(fr, arr, label=lbl)
                    legF.append(lbl)
                elif ext == '.csv' and prm == 's11':
                    df = pd.read_csv(p)
                    fr = df.iloc[:,0].values
                    arr = df.iloc[:,1].values
                    axF.plot(fr, arr, label=lbl)
                    legF.append(lbl)
            except Exception:
                pass
        axF.set_ylabel(f"|{prm.upper()}| [dB]")
        axF.set_title(f"{prm.upper()} (frequency domain - raw)")
        axF.grid(True)
        if legF:
            axF.legend(loc="upper right")
        else:
            axF.text(0.5, 0.5, "Brak danych", ha="center", va="center", color="gray")
            axF.set_xticks([]); axF.set_yticks([])
        # time domain raw
        for v, p, d in self.fls:
            if not v.get(): continue
            ext = Path(p).suffix.lower()
            lbl = Path(p).stem
            try:
                if ext in ['.s1p', '.s2p', '.s3p']:
                    ntw = loadFile(p)
                    getattr(ntw, prm).plot_s_db_time(ax=axTR, label=lbl)
                    legTR.append(lbl)
            except Exception:
                pass
        axTR.set_ylabel(f"{prm.upper()} TD [dB]")
        axTR.set_xlabel("Czas [ns]")
        axTR.set_title(f"{prm.upper()} (time domain - raw)")
        axTR.grid(True)
        axTR.set_xlim(0, 50)
        if legTR:
            axTR.legend(loc="upper right")
        else:
            axTR.plot([0, 30], [0, 0], alpha=0)
            axTR.text(0.5, 0.5, "Brak danych / Placeholder", ha="center", va="center", color="gray", transform=axTR.transAxes)
            axTR.set_xticks(np.linspace(0, 30, 7))
            axTR.set_yticks([])
        # freq domain gated
        if doGate:
            for v, p, d in self.fls:
                if not v.get(): continue
                ext = Path(p).suffix.lower()
                lbl = Path(p).stem
                try:
                    if ext in ['.s1p', '.s2p', '.s3p']:
                        ntw = loadFile(p)
                        sRaw = getattr(ntw, prm)
                        sGate = sRaw.time_gate(center=center, span=span)
                        # wróć do freq domain gated
                        freq = ntw.f
                        arr = sGate.s_db.flatten()
                        axTG.plot(freq, arr, label=lbl)
                        legTG.append(lbl)
                except Exception:
                    pass
        axTG.set_ylabel(f"{prm.upper()} [dB]")
        axTG.set_xlabel("Frequency [Hz]")
        axTG.set_title(f"{prm.upper()} (frequency domain - gated)")
        axTG.grid(True)
        if legTG:
            axTG.legend(loc="upper right")
        else:
            axTG.text(0.5, 0.5, "Brak danych / Placeholder", ha="center", va="center", color="gray", transform=axTG.transAxes)
            axTG.set_xticks([]); axTG.set_yticks([])
        self.figT.tight_layout()
        self.cvT.draw()

    def _clearM(self, upd=False):
        for m in self.mrk:
            try: 
                m['marker'].remove()
                m['text'].remove()
            except: 
                pass
        self.mrk.clear()
        if not upd:
            self.mtxt.clear()
            self.txt.config(state=tk.NORMAL)
            self.txt.delete(1.0, tk.END)
            self.txt.config(state=tk.DISABLED)
        self.cv.draw()
        self.cvT.draw()

    def _updAll(self):
        self._updPlot()
        self._updTPlot()
        if self.tab == "regex":
            self._updRegexPlot()
        elif self.tab == "overlap":
            self._updOverlapPlot()
        elif self.tab == "variance":
            self._updVariancePlot()

    def _onClick(self, ev):
        if getattr(self, "_blkNext", False):
            self._blkNext = False
            return
        if self.tab == "freq":
            axes = self.fax
        elif self.tab == "time":
            axes = self.tax
        else:
            return
        if self._isOnM(ev, axes):
            return
        axc = None
        for ax in axes:
            if ev.inaxes == ax:
                axc = ax
                break
        if axc is None: return
        x = ev.xdata
        y = ev.ydata
        if x is None or y is None: return
        mind = float("inf")
        best = None
        for ln in axc.get_lines():
            xd = ln.get_xdata()
            yd = ln.get_ydata()
            dists = np.sqrt((xd - x) ** 2 + (yd - y) ** 2)
            idx = np.argmin(dists)
            if dists[idx] < mind:
                mind = dists[idx]
                xf = xd[idx]
                yf = yd[idx]
                lbl = ln.get_label()
                col = ln.get_color()
                best = (xf, yf, lbl, col)
        if best:
            xf, yf, lbl, col = best
            m, = axc.plot([xf], [yf], 'o', color=col, markersize=9, picker=7)
            if self.tab == "time":
                if xf < 1e-6:
                    xf_ns = xf * 1e9
                    t = axc.annotate(
                        f"{round(xf_ns,3)} ns\n{round(yf,2)} dB",
                        xy=(xf, yf), xytext=(10, 20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.5),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
                        fontsize=9
                    )
                    txt = f"{lbl}: {round(xf_ns,3)} ns  |  {round(yf,2)} dB\n"
                else:
                    t = axc.annotate(
                        f"{round(xf,3)} ns\n{round(yf,2)} dB",
                        xy=(xf, yf), xytext=(10, 20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.5),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
                        fontsize=9
                    )
                    txt = f"{lbl}: {round(xf,3)} ns  |  {round(yf,2)} dB\n"
            else:
                t = axc.annotate(
                    f"{round(xf/1e6,3)} MHz\n{round(yf,2)} dB",
                    xy=(xf, yf), xytext=(10, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.5),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
                    fontsize=9
                )
                txt = f"{lbl}: {round(xf/1e6,3)} MHz  |  {round(yf,2)} dB\n"
            t.set_picker(True)
            self.mrk.append({'marker': m, 'text': t, 'ax': axc})
            self.mtxt.append(txt)
            self.txt.config(state=tk.NORMAL)
            self.txt.insert(tk.END, txt)
            self.txt.config(state=tk.DISABLED)
            if self.tab == "freq":
                self.cv.draw()
            else:
                self.cvT.draw()

    def _isOnM(self, ev, axes):
        for ent in self.mrk:
            if ent['ax'] not in axes:
                continue
            ann = ent['text']
            bbox = ann.get_window_extent()
            x, y = ev.x, ev.y
            if bbox.contains(x, y):
                return True
        return False

    def _onPick(self, ev):
        for ent in self.mrk:
            if ev.artist is ent['text']:
                ent['marker'].remove()
                ent['text'].remove()
                self.mrk.remove(ent)
                self._blkNext = True
                if self.tab == "freq":
                    self.cv.draw()
                else:
                    self.cvT.draw()
                break

    def _exportRanges(self):
        if not self.regexRanges:
            tk.messagebox.showinfo("No ranges", "No monotonic frequency ranges found.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save frequency ranges",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(f"# Monotonic frequency ranges for pattern: {self.regexPattern.get()}\n")
                f.write(f"# S-parameter: {self.regexParam.get().upper()}\n")
                mode = "Strict monotonic" if self.regexStrictMonotonic.get() else "Non-strict monotonic"
                f.write(f"# Monotonicity mode: {mode}\n")
                f.write(f"# Format: start_freq end_freq (Hz)\n")
                f.write("#\n")
                for start, end in self.regexRanges:
                    f.write(f"{start} {end}\n")
                f.write("\n# Ranges in GHz:\n")
                for start, end in self.regexRanges:
                    f.write(f"{start/1e9:.3f} {end/1e9:.3f}\n")
            tk.messagebox.showinfo("Export complete", f"Ranges saved to {filename}")

    def _deleteSelectedFiles(self):
        toDelete = []
        for i, (v, p, d) in enumerate(self.fls):
            if v.get():
                toDelete.append(i)
        
        if not toDelete:
            messagebox.showinfo("No selection", "No files selected for deletion")
            return
            
        if messagebox.askyesno("Confirm deletion", f"Remove {len(toDelete)} selected files from the list?"):
            for i in reversed(toDelete):
                self.fls.pop(i)
            
            for widget in self.fbox.winfo_children():
                widget.destroy()
            
            for v, p, d in self.fls:
                chk = ttk.Checkbutton(self.fbox, text=Path(p).name, variable=v, command=self._updAll)
                chk.pack(anchor="w")
            
            self.fileListCanvas.configure(scrollregion=self.fileListCanvas.bbox("all"))
            self._updAll()
            messagebox.showinfo("Files removed", f"Removed {len(toDelete)} files from the list")

    def _parseFrequencyFile(self, filepath):
        ranges = []
        with open(filepath, 'r') as file:
            for line in file:
                match = re.match(r"^\s*(\d+\.?\d*)\s+(\d+\.?\d*)", line)
                if match:
                    start = float(match.group(1)) / 1e9  
                    end = float(match.group(2)) / 1e9
                    ranges.append((start, end))
        return ranges

    def _loadRangeFiles(self):
        files = filedialog.askopenfilenames(
            title="Select frequency range files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not files:
            return
            
        self.overlapData = {}
        for filepath in files:
            key = Path(filepath).stem.split('_')[0]
            self.overlapData[key] = self._parseFrequencyFile(filepath)
        
        self._updOverlapPlot()

    def _analyzeFromRegex(self):
        if not self.regexRanges:
            messagebox.showinfo("No data", "No regex ranges available. Run regex analysis first.")
            return
            
        key = f"regex_{self.regexParam.get()}"
        self.overlapData[key] = [(s/1e9, e/1e9) for s, e in self.regexRanges]  # Convert to GHz
        self._updOverlapPlot()

    def _findOverlaps(self, r1, r2):
        overlaps = []
        for s1, e1 in r1:
            for s2, e2 in r2:
                start = max(s1, s2)
                end = min(e1, e2)
                if start <= end:
                    overlaps.append((start, end))
        return overlaps

    def _updOverlapPlot(self):
        self.figO.clear()
        self.axO = self.figO.add_subplot(111)
        
        if not self.overlapData:
            self.axO.text(0.5, 0.5, "Load frequency range files to visualize overlaps", 
                         ha="center", va="center", fontsize=12, color="gray")
            self.axO.set_xticks([])
            self.axO.set_yticks([])
            self.cvO.draw()
            return
        
        colors = cycle(['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#17becf'])
        y_positions = {}
        legend_elements = []
        
        for i, (name, ranges) in enumerate(sorted(self.overlapData.items())):
            color = next(colors)
            y_positions[name] = i
            for start, end in ranges:
                self.axO.plot([start, end], [i, i], color=color, linewidth=4)
            legend_elements.append(mpatches.Patch(color=color, label=f"{name}"))
        
        for (name1, r1), (name2, r2) in combinations(self.overlapData.items(), 2):
            overlaps = self._findOverlaps(r1, r2)
            for s, e in overlaps:
                y1, y2 = y_positions[name1], y_positions[name2]
                ymin, ymax = sorted([y1, y2])
                self.axO.fill_betweenx([ymin - 0.15, ymax + 0.15], s, e, color='red', alpha=0.3)
        
        self.txt.config(state=tk.NORMAL)
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, "Frequency ranges (GHz):\n")
        self.txt.insert(tk.END, "-" * 40 + "\n")
        for name, ranges in sorted(self.overlapData.items()):
            formatted = [(round(s, 3), round(e, 3)) for s, e in ranges]
            self.txt.insert(tk.END, f"{name}: {formatted}\n")
        
        self.txt.insert(tk.END, "\n📊 Overlapping ranges:\n")
        self.txt.insert(tk.END, "-" * 40 + "\n")
        for (n1, r1), (n2, r2) in combinations(self.overlapData.items(), 2):
            overlaps = self._findOverlaps(r1, r2)
            if overlaps:
                formatted = [(round(s, 3), round(e, 3)) for s, e in overlaps]
                self.txt.insert(tk.END, f"{n1} ↔ {n2}: {formatted}\n")
            else:
                self.txt.insert(tk.END, f"{n1} ↔ {n2}: no overlaps\n")
        self.txt.config(state=tk.DISABLED)
        
        self.axO.set_yticks(list(y_positions.values()))
        ylabels = list(y_positions.keys())
        if len(ylabels) > 15:
            self.axO.set_yticklabels(ylabels, fontsize=8)
        else:
            self.axO.set_yticklabels(ylabels)
        self.axO.set_xlabel("Frequency (GHz)")
        self.axO.set_title("Frequency ranges and overlaps")
        self.axO.grid(True, linestyle='--', alpha=0.5)
        
        if len(legend_elements) <= 10:
            self.axO.legend(handles=legend_elements + [mpatches.Patch(color='red', alpha=0.3, label='Overlaps')])
        
        self.figO.tight_layout()
        self.cvO.draw()

    def _exportOverlaps(self):
        if not self.overlapData:
            messagebox.showinfo("No data", "No overlap data to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save overlap analysis",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Frequency ranges (GHz):\n\n")
            for name, ranges in sorted(self.overlapData.items()):
                formatted = [(round(s, 3), round(e, 3)) for s, e in ranges]
                f.write(f"{name}: {formatted}\n")
            
            f.write("\n📊 Overlapping ranges:\n\n")
            for (n1, r1), (n2, r2) in combinations(self.overlapData.items(), 2):
                overlaps = self._findOverlaps(r1, r2)
                if overlaps:
                    formatted = [(round(s, 3), round(e, 3)) for s, e in overlaps]
                    f.write(f"{n1} ↔ {n2}: {formatted}\n")
                else:
                    f.write(f"{n1} ↔ {n2}: no overlaps\n")
        
        messagebox.showinfo("Export complete", f"Overlap analysis saved to {filename}")

    def _clearOverlapPlot(self):
        self.overlapData = {}
        self._updOverlapPlot()

    def _normalizeForVariance(self, data, is_phase=False):
        """Normalize data for variance analysis following basic_stats.py approach"""
        if is_phase and self.varDetrend.get():
            # For phase: detrend then z-score normalize
            n_samples, n_frequencies = data.shape
            detrended = np.zeros_like(data)
            
            for i in range(n_samples):
                # Fit linear trend
                x = np.arange(n_frequencies)
                coeffs = np.polyfit(x, data[i], 1)
                trend = np.polyval(coeffs, x)
                detrended[i] = data[i] - trend
            
            # Z-score normalize the detrended data
            mean = np.mean(detrended)
            std = np.std(detrended) + 1e-10
            return (detrended - mean) / std
        else:
            # For magnitude or phase without detrending: standard z-score
            mean = np.mean(data)
            std = np.std(data) + 1e-10
            return (data - mean) / std

    def _calculateVariance(self):
        fmin = self.fmin.get()
        fmax = self.fmax.get()
        sstr = f"{fmin}-{fmax}ghz"
        
        param_list = []
        component_types = []  # Track which are magnitude vs phase
        
        if self.varS11.get():
            if self.varMag.get():
                param_list.append('s11_mag')
                component_types.append('mag')
            if self.varPhase.get():
                param_list.append('s11_phase')
                component_types.append('phase')
        if self.varS12.get():
            if self.varMag.get():
                param_list.append('s12_mag')
                component_types.append('mag')
            if self.varPhase.get():
                param_list.append('s12_phase')
                component_types.append('phase')
        if self.varS21.get():
            if self.varMag.get():
                param_list.append('s21_mag')
                component_types.append('mag')
            if self.varPhase.get():
                param_list.append('s21_phase')
                component_types.append('phase')
        if self.varS22.get():
            if self.varMag.get():
                param_list.append('s22_mag')
                component_types.append('mag')
            if self.varPhase.get():
                param_list.append('s22_phase')
                component_types.append('phase')
        
        if not param_list:
            return None
        
        # Collect raw data first
        raw_data_by_param = {param: [] for param in param_list}
        file_count = 0
        freq_ref = None
        
        for v, p, d in self.fls:
            if not v.get():
                continue
                
            ext = Path(p).suffix.lower()
            if ext not in ['.s1p', '.s2p', '.s3p']:
                continue
                
            try:
                ntw = loadFile(p)[sstr]
                if freq_ref is None:
                    freq_ref = ntw.f
                elif len(ntw.f) != len(freq_ref) or not np.allclose(ntw.f, freq_ref, rtol=1e-6):
                    continue
                
                # Collect data for each S-parameter
                for sparam in ['s11', 's12', 's21', 's22']:
                    if any(sparam in p for p in param_list):
                        s_data = getattr(ntw, sparam)
                        
                        if f'{sparam}_mag' in param_list:
                            raw_data_by_param[f'{sparam}_mag'].append(s_data.s_db.flatten())
                        
                        if f'{sparam}_phase' in param_list:
                            phase_rad = np.angle(s_data.s.flatten())
                            # Check if phase needs unwrapping (large jumps indicate wrapping)
                            phase_deg = np.degrees(phase_rad)
                            if np.any(np.abs(np.diff(phase_deg)) > 180):
                                phase_unwrapped_rad = np.unwrap(phase_rad)
                                phase_deg = np.degrees(phase_unwrapped_rad)
                            raw_data_by_param[f'{sparam}_phase'].append(phase_deg)
                
                file_count += 1
                    
            except Exception as e:
                pass
        
        if file_count < 2 or freq_ref is None:
            return None
        
        n_frequencies = len(freq_ref)
        n_params = len(param_list)
        
        # Normalize each parameter type separately
        normalized_data = np.zeros((file_count, n_frequencies, n_params))
        
        for i, (param, comp_type) in enumerate(zip(param_list, component_types)):
            # Convert list to array: (samples, frequencies)
            param_data = np.array(raw_data_by_param[param])
            
            # Normalize based on type
            normalized = self._normalizeForVariance(param_data, is_phase=(comp_type == 'phase'))
            normalized_data[:, :, i] = normalized
        
        # Calculate variance on normalized data
        variance_by_param = np.var(normalized_data, axis=0, ddof=1)
        total_variance = np.sum(variance_by_param, axis=1)
        
        # Calculate percentage contribution
        variance_contribution = np.zeros_like(variance_by_param)
        for freq_idx in range(n_frequencies):
            if total_variance[freq_idx] > 1e-10:
                variance_contribution[freq_idx] = (variance_by_param[freq_idx] / 
                                                 total_variance[freq_idx]) * 100
        
        # Stats on normalized data
        mean_by_param = np.mean(normalized_data, axis=0)
        std_by_param = np.std(normalized_data, axis=0, ddof=1)
        
        # Create normalization info string
        if self.varDetrend.get():
            norm_info = 'Phase: detrended+z-score, Magnitude: z-score'
        else:
            norm_info = 'Phase: z-score (no detrending), Magnitude: z-score'
        
        return {
            'frequencies': freq_ref,
            'variance_by_param': variance_by_param,
            'variance_contribution': variance_contribution,
            'total_variance': total_variance,
            'mean_by_param': mean_by_param,
            'std_by_param': std_by_param,
            'n_params': n_params,
            'param_list': param_list,
            'file_count': file_count,
            'normalized_data': normalized_data,
            'normalization_info': norm_info
        }

    def _updVariancePlot(self):
        self.figV.clear()
        self.axV = self.figV.add_subplot(111)
        
        self.varData = self._calculateVariance()
        
        if self.varData is None:
            self.axV.text(0.5, 0.5, "Select S-parameters and at least 2 files", 
                         ha="center", va="center", fontsize=12, color="gray")
            self.axV.set_xticks([])
            self.axV.set_yticks([])
            self.cvV.draw()
            return
        
        freqs = self.varData['frequencies']
        var_contrib = self.varData['variance_contribution']
        n_params = self.varData['n_params']
        
        param_labels = []
        for param in self.varData['param_list']:
            # param is already like 's11_mag' or 's11_phase'
            parts = param.split('_')
            s_param = parts[0].upper()
            comp_type = parts[1].capitalize()
            param_labels.append(f'{s_param} {comp_type}')
        
        y = np.zeros((n_params, len(freqs)))
        for i in range(n_params):
            y[i] = var_contrib[:, i]
        
        # Use distinct colors for better visibility
        if n_params <= 8:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'][:n_params]
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_params))
        
        self.axV.stackplot(freqs, y, labels=param_labels[:n_params], 
                          colors=colors, alpha=0.8)
        
        # Mark high variance frequencies
        mean_var = np.mean(self.varData['total_variance'])
        std_var = np.std(self.varData['total_variance'])
        high_var_mask = self.varData['total_variance'] > mean_var + 2 * std_var
        
        if np.any(high_var_mask):
            high_var_freqs = freqs[high_var_mask]
            for f in high_var_freqs:
                self.axV.axvline(x=f, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        self.axV.set_xlabel('Frequency [Hz]')
        self.axV.set_ylabel('Variance Contribution (%)')
        self.axV.set_title(f'Normalized Variance Contribution by Component ({self.varData["file_count"]} files)')
        self.axV.grid(True, alpha=0.3)
        self.axV.set_ylim(0, 105)  # Give a bit more space at the top
        
        if n_params <= 8:
            self.axV.legend(loc='upper right', fontsize=9)
        
        # Store data for markers
        self.varFreqs = freqs
        self.varTotalVar = self.varData['total_variance']
        
        # Update text widget
        mean_var = np.mean(self.varData['total_variance'])
        std_var = np.std(self.varData['total_variance'])
        self.txt.config(state=tk.NORMAL)
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, f"Variance Statistics (Normalized Data):\n")
        self.txt.insert(tk.END, f"Normalization: {self.varData['normalization_info']}\n")
        self.txt.insert(tk.END, f"Mean total variance: {mean_var:.6f}\n")
        self.txt.insert(tk.END, f"Std of variance: {std_var:.6f}\n")
        self.txt.insert(tk.END, f"Files analyzed: {self.varData['file_count']}\n")
        self.txt.insert(tk.END, "-" * 40 + "\n")
        for txt in self.varMtxt:
            self.txt.insert(tk.END, txt)
        self.txt.config(state=tk.DISABLED)
        
        # Adjust layout to prevent marker cutoff
        self.figV.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top
        self.cvV.draw()

    def _onMotionVar(self, event):
        if event.inaxes != self.axV or self.varData is None:
            return
            
        x = event.xdata
        if x is None:
            return
            
        idx = np.argmin(np.abs(self.varFreqs - x))
        freq = self.varFreqs[idx]
        var = self.varTotalVar[idx]
        
        var_contrib = self.varData['variance_contribution'][idx, :]
        
        msg = f"Freq: {freq/1e6:.3f} MHz, Total Var: {var:.6f}"
        
        if len(var_contrib) > 0:
            top_idx = np.argmax(var_contrib)
            param_labels = []
            for param in self.varData['param_list']:
                parts = param.split('_')
                s_param = parts[0].upper()
                comp_type = parts[1].capitalize()
                param_labels.append(f'{s_param} {comp_type}')
            
            if top_idx < len(param_labels):
                msg += f", Max: {param_labels[top_idx]} ({var_contrib[top_idx]:.1f}%)"
        
        self.tbV.set_message(msg)

    def _onClickVar(self, event):
        if getattr(self, "_blkNextVar", False):
            self._blkNextVar = False
            return
            
        if event.inaxes != self.axV or self.varData is None:
            return
            
        x = event.xdata
        if x is None:
            return
            
        idx = np.argmin(np.abs(self.varFreqs - x))
        freq = self.varFreqs[idx]
        var = self.varTotalVar[idx]
        
        # Place marker slightly below 100% to ensure visibility
        y_pos = 95  # Fixed position at 95%
        
        m, = self.axV.plot([freq], [y_pos], 'ro', markersize=10, picker=7)
        
        # Add annotation with downward pointing arrow
        t = self.axV.annotate(
            f"{freq/1e6:.3f} MHz\nVar: {var:.4f}",
            xy=(freq, y_pos), xytext=(10, -30),  # Place text below marker
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
            fontsize=9
        )
        t.set_picker(True)
        
        self.varMrk.append({'marker': m, 'text': t})
        
        # Add detailed breakdown to text
        txt = f"Variance marker: {freq/1e6:.3f} MHz\n"
        txt += f"  Total variance: {var:.4f}\n"
        
        # Add breakdown by parameter
        param_labels = []
        for param in self.varData['param_list']:
            parts = param.split('_')
            s_param = parts[0].upper()
            comp_type = parts[1].capitalize()
            param_labels.append(f'{s_param} {comp_type}')
        
        var_by_param = self.varData['variance_by_param'][idx]
        var_contrib = self.varData['variance_contribution'][idx]
        
        for i, label in enumerate(param_labels):
            txt += f"  {label}: {var_by_param[i]:.6f} ({var_contrib[i]:.1f}%)\n"
        
        self.varMtxt.append(txt)
        self.txt.config(state=tk.NORMAL)
        self.txt.insert(tk.END, txt)
        self.txt.config(state=tk.DISABLED)
        
        self.cvV.draw()

    def _onPickVar(self, event):
        for ent in self.varMrk:
            if event.artist is ent['text']:
                ent['marker'].remove()
                ent['text'].remove()
                self.varMrk.remove(ent)
                self._blkNextVar = True
                self.cvV.draw()
                break

    def _clearVarMarkers(self):
        for m in self.varMrk:
            try:
                m['marker'].remove()
                m['text'].remove()
            except:
                pass
        self.varMrk.clear()
        self.varMtxt.clear()
        
        if self.varData:
            mean_var = np.mean(self.varData['total_variance'])
            std_var = np.std(self.varData['total_variance'])
            self.txt.config(state=tk.NORMAL)
            self.txt.delete(1.0, tk.END)
            self.txt.insert(tk.END, f"Variance Statistics (Normalized Data):\n")
            self.txt.insert(tk.END, f"Normalization: {self.varData['normalization_info']}\n")
            self.txt.insert(tk.END, f"Mean total variance: {mean_var:.6f}\n")
            self.txt.insert(tk.END, f"Std of variance: {std_var:.6f}\n")
            self.txt.insert(tk.END, f"Files analyzed: {self.varData['file_count']}\n")
            self.txt.insert(tk.END, "-" * 40 + "\n")
            self.txt.config(state=tk.DISABLED)
        else:
            self.txt.config(state=tk.NORMAL)
            self.txt.delete(1.0, tk.END)
            self.txt.config(state=tk.DISABLED)
            
        self.cvV.draw()

    def _exportVariance(self):
        if self.varData is None:
            messagebox.showinfo("No data", "No variance data to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save variance data",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        with open(filename, 'w') as f:
            f.write("# Variance Analysis (Normalized Data)\n")
            f.write(f"# Normalization: {self.varData['normalization_info']}\n")
            f.write(f"# Parameters: {', '.join(self.varData['param_list'])}\n")
            f.write(f"# Number of files: {self.varData['file_count']}\n")
            f.write(f"# Mean total variance: {np.mean(self.varData['total_variance']):.6f}\n")
            f.write(f"# Std total variance: {np.std(self.varData['total_variance']):.6f}\n")
            f.write("# Frequency[Hz] Total_Variance Component_Variances...\n")
            
            for i, freq in enumerate(self.varData['frequencies']):
                f.write(f"{freq} {self.varData['total_variance'][i]}")
                for j in range(self.varData['n_params']):
                    f.write(f" {self.varData['variance_by_param'][i, j]}")
                f.write("\n")
        
        messagebox.showinfo("Export complete", f"Variance data saved to {filename}")

if __name__ == "__main__":
    App().mainloop()
