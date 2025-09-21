import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import os
import threading
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as mpatches
import skrf as rf
import pandas as pd
import tempfile
import numpy as np
import re
from itertools import combinations, cycle
from scipy import signal
from scipy.ndimage import convolve1d

# TODO: 0. FUCKING MATPLOTLIB CHRIST ALMIGHTY
# TODO: 1. caching
# DONE: 2. fix importing to regex ie. no overwriting
# DONE: 3. replace frechet with crosscorelation?
# TODO: 4. replace file loading to c
# DUMB: 5. SIMD based optimalizations? no, numpy is already optimized and we're not doing live streaming of sparams data
# DONE?: 6. integrate sparameters curves
# DONE: 7. Make new submenu for shape / maybe stats?
# TODO: 8. add basic stats per sparameter file
# DONE: 9. L1 L2 Lwhat have you for shape similarity / general similarity
# TODO: 10. throw ml into separate thing no one save for mc will use it
# TODO: 11. christ but is matplotlib slow as sin and is it used badly here fix it
# TODO: 12. file parsing is bad no good fix that
# TODO: 13. numba jit potentially
# TODO: 14. might have to rewrite parts in c
# DONE: 15. averages of sparams
# TODO: make averages have different colors
# TODO: make naming averages not be a major pain in the ass
# TODO: add local extrema calculation/marking on plots
# TODO: add linear option to all plots i suppose
# TODO: maybe make the integration not be retarded - space between curves pairwise



if rf.__version__ > "0.17.0":
    def find_nearest_index(array, value):
        return (np.abs(array-value)).argmin()

    def delay_v017(self, d, unit='deg', port=0, media=None, **kw):
        if d == 0: return self
        d = d/2.
        if self.nports > 2:
            raise NotImplementedError('only implemented for 1 and 2 ports')
        if media is None:
            try:
                from skrf.media import Freespace
            except:
                from skrf.media.freespace import Freespace
            media = Freespace(frequency=self.frequency, z0=self.z0[:,port])
        l = media.line(d=d, unit=unit, **kw)
        return l**self

    def time_gate_v017(self, center=None, span=None, **kwargs):
        if self.nports > 1:
            raise ValueError('Time-gating only works on one-ports')

        window = kwargs.get('window', ('kaiser', 6))
        mode = kwargs.get('mode', 'bandpass')
        boundary = kwargs.get('boundary', 'reflect')
        media = kwargs.get('media', None)

        center_s = center * 1e-9
        span_s = span * 1e-9
        start = center_s - span_s/2.
        stop = center_s + span_s/2.

        t = np.linspace(-0.5/self.frequency.step, 0.5/self.frequency.step, 
                        self.frequency.npoints)

        start_idx = find_nearest_index(t, start)
        stop_idx = find_nearest_index(t, stop)
        window_width = abs(stop_idx - start_idx)

        if window_width == 0:
            window_array = np.array([])
        else:
            window_array = signal.get_window(window, window_width)

        gate = np.r_[np.zeros(start_idx),
                     window_array,
                     np.zeros(len(t) - stop_idx)]

        kernel = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(gate, axes=0), axis=0))
        kernel = abs(kernel).flatten()
        kernel = kernel/sum(kernel)

        out = self.copy()

        if center != 0:
            out = delay_v017(out, -center, 'ns', port=0, media=media)

        re = np.real(out.s[:,0,0])
        im = np.imag(out.s[:,0,0])
        s = convolve1d(re, kernel, mode=boundary) + \
            1j*convolve1d(im, kernel, mode=boundary)
        out.s[:,0,0] = s

        if center != 0:
            out = delay_v017(out, center, 'ns', port=0, media=media)

        if mode == 'bandstop':
            out = self - out

        return out

    rf.Network.delay = delay_v017
    rf.Network.time_gate = time_gate_v017

MAXF = 4
MINF = 0.4


def loadFile(p):
    print(f"loadFile called with: {p}")
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    l = None
    
    for encoding in encodings:
        try:
            with open(p, "r", encoding=encoding) as f:
                l = f.readlines()
                break
        except UnicodeDecodeError as e:
            continue
    
    if l is None:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
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
    
    try:
        network = rf.Network(t.name)
        return network
    except Exception as e:
        raise


class ValidatedDoubleVar(tk.DoubleVar):
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
        self.regexStrictMonotonic = tk.BooleanVar(value=False)
        self.regexGateChk = tk.BooleanVar(value=False)
        self.regexGroup = tk.IntVar(value=1)
        self.regexSpans = []
        self.regexLines = []
        self.regexRanges = []
        self.regexSmallDiffChk = tk.BooleanVar(value=False)
        self.regexSmallDiffThreshold = ValidatedDoubleVar(value=0.1)
        self.regexSmallDiffMinCount = tk.IntVar(value=1)
        self.regexSmallDiffRefDB = ValidatedDoubleVar(value=10.0)
        
        self.overlapData = {}
        self.fileListCanvas = None
        
        self.varS11 = tk.BooleanVar(value=True)
        self.varS22 = tk.BooleanVar(value=False)
        self.varS12 = tk.BooleanVar(value=False)
        self.varS21 = tk.BooleanVar(value=True)
        self.varMag = tk.BooleanVar(value=True)
        self.varPhase = tk.BooleanVar(value=True)
        self.varDetrend = tk.BooleanVar(value=True)
        self.varData = None
        self.varMrk = []
        self.varMtxt = []
        self._blkNextVar = False
        
        self.trainDataDirs = []
        self.trainHiddenDim = tk.IntVar(value=256)
        self.trainLatentDim = tk.IntVar(value=64)
        self.trainDropout = tk.DoubleVar(value=0.1)
        self.trainLR = tk.DoubleVar(value=0.001)
        self.trainWeightDecay = tk.DoubleVar(value=0.001)
        self.trainEpochs = tk.IntVar(value=200)
        self.trainPatience = tk.IntVar(value=25)
        self.trainLossFn = tk.StringVar(value="mse")
        self.trainAugment = tk.BooleanVar(value=False)
        self.trainNoise = tk.DoubleVar(value=0.05)
        self.trainBatchNorm = tk.BooleanVar(value=True)
        self.trainDevice = tk.StringVar(value="cpu")
        self.modelTrainingAvailable = False
        
        self.markersEnabled = tk.BooleanVar(value=False)
        self.markersEnabledTime = tk.BooleanVar(value=False)

        self.extremaEnabled = tk.BooleanVar(value=False)
        self.extremaMinima = tk.BooleanVar(value=True)
        self.extremaMaxima = tk.BooleanVar(value=True)
        self.extremaRangeMin = ValidatedDoubleVar(value=0.0)
        self.extremaRangeMax = ValidatedDoubleVar(value=10.0)
        self.extremaLines = []
        self.extremaLinesT = []
        self.extremaLinesR = []

        self.tdAnalysisData = None
        self.tdAnalysisMrk = []
        self.tdAnalysisMtxt = []
        self.crossCorrData = None

        self.maxLag = tk.IntVar(value=5)
        self.avgCounter = 0

        self.shapeAdaptive = tk.BooleanVar(value=False)
        self.shapeAlpha = tk.DoubleVar(value=2.0)
        self.shapeSigma = tk.IntVar(value=5)
        self.shapeGamma = tk.DoubleVar(value=2.0)
        self.shapeWmin = tk.DoubleVar(value=0.1)
        self.shapeActivityType = tk.StringVar(value="nonincreasing")
        self.shapeShowActivity = tk.BooleanVar(value=False)
                
        try:
            import water_pred
            import torch
            self.modelTrainingAvailable = True
            self.water_pred = water_pred
            self.torch = torch
            if torch.cuda.is_available():
                self.trainDevice.set("cuda")
        except ImportError:
            self.water_pred = None
            self.torch = None
        
        self._makeUi()
        self._updAll()

    def _validateNumeric(self, widget, var, callback=None):
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
        btn4 = ttk.Button(lfrm, text="Average files", command=self._avgFiles)
        btn4.pack(anchor="w", pady=(0,10))
        
        self.legendVisible = tk.BooleanVar(value=True)

        self.legendOnPlot = tk.BooleanVar(value=False)

        ttk.Checkbutton(lfrm, text="Show legend panel", variable=self.legendVisible, 
                       command=self._toggleLegend).pack(anchor="w", pady=(0,10))

        ttk.Checkbutton(lfrm, text="Legend on plot", variable=self.legendOnPlot,
                       command=self._updAll).pack(anchor="w", pady=(0,10))

        
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
        ttk.Checkbutton(self.cbox, text="Enable markers", variable=self.markersEnabled).pack(side=tk.LEFT, padx=(20,2))
        self.extremaBox = ttk.Frame(lfrm)
        ttk.Checkbutton(self.extremaBox, text="Show extrema", variable=self.extremaEnabled, command=self._updAll).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.extremaBox, text="Min", variable=self.extremaMinima, command=self._updAll).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.extremaBox, text="Max", variable=self.extremaMaxima, command=self._updAll).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.extremaBox, text="Range [GHz]:").pack(side=tk.LEFT, padx=(10,2))
        rangeMinEntry = ttk.Entry(self.extremaBox, textvariable=self.extremaRangeMin, width=5)
        rangeMinEntry.pack(side=tk.LEFT, padx=2)
        self._validateNumeric(rangeMinEntry, self.extremaRangeMin, self._updAll)
        ttk.Label(self.extremaBox, text="-").pack(side=tk.LEFT, padx=2)
        rangeMaxEntry = ttk.Entry(self.extremaBox, textvariable=self.extremaRangeMax, width=5)
        rangeMaxEntry.pack(side=tk.LEFT, padx=2)
        self._validateNumeric(rangeMaxEntry, self.extremaRangeMax, self._updAll)
        self.rbox = ttk.Frame(lfrm)
        ttk.Label(self.rbox, text="Wybierz TDG:").pack(side=tk.LEFT)
        for n in ("s11", "s12", "s21", "s22"):
            rb = ttk.Radiobutton(self.rbox, text=n.upper(), value=n, variable=self.td, command=self._updAll)
            rb.pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.rbox, text="Enable markers", variable=self.markersEnabledTime).pack(side=tk.LEFT, padx=(20,2))
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
        
        self.regexBox = ttk.Frame(lfrm)
        self.regexPhaseChk = tk.BooleanVar(value=False)

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
        
        smallDiffFrame = ttk.Frame(self.regexBox)
        smallDiffFrame.pack(anchor="w", pady=(5,0))
        self.regexSmallDiffCheckBtn = ttk.Checkbutton(smallDiffFrame, text="Highlight small diffs", 
                                                      variable=self.regexSmallDiffChk, 
                                                      command=self._updRegexPlot)
        self.regexSmallDiffCheckBtn.pack(side=tk.LEFT, padx=3)
        ttk.Label(smallDiffFrame, text="Tolerance:").pack(side=tk.LEFT, padx=(10,2))
        thresholdEntry = ttk.Entry(smallDiffFrame, textvariable=self.regexSmallDiffThreshold, width=6)
        thresholdEntry.pack(side=tk.LEFT, padx=2)
        self._validateNumeric(thresholdEntry, self.regexSmallDiffThreshold, self._updRegexPlot)
        ttk.Label(smallDiffFrame, text="@").pack(side=tk.LEFT, padx=(2,2))
        refDBEntry = ttk.Entry(smallDiffFrame, textvariable=self.regexSmallDiffRefDB, width=6)
        refDBEntry.pack(side=tk.LEFT, padx=2)
        self._validateNumeric(refDBEntry, self.regexSmallDiffRefDB, self._updRegexPlot)
        ttk.Label(smallDiffFrame, text="dB ref").pack(side=tk.LEFT, padx=(0,5))
        ttk.Label(smallDiffFrame, text="Min count:").pack(side=tk.LEFT, padx=(10,2))
        minCountSpin = ttk.Spinbox(smallDiffFrame, from_=1, to=10, width=5,
                                   textvariable=self.regexSmallDiffMinCount, 
                                   command=self._updRegexPlot)
        minCountSpin.pack(side=tk.LEFT, padx=2)
        ttk.Label(smallDiffFrame, text="diffs", font=("", 8)).pack(side=tk.LEFT, padx=(0,5))
        
        strictHelpFrame = ttk.Frame(self.regexBox)
        strictHelpFrame.pack(anchor="w", pady=(2,0))
        ttk.Label(strictHelpFrame, text="Strict: no equal consecutive values; Non-strict: allows equal values", 
                 font=("", 8), foreground="gray").pack(anchor="w")
        
        self.colorInfoFrame = ttk.Frame(self.regexBox)
        self.colorInfoFrame.pack(anchor="w", pady=(2,0))
        self.colorInfoLabel = ttk.Label(self.colorInfoFrame, text="", font=("", 8), foreground="gray")
        self.colorInfoLabel.pack(anchor="w")
        self._updateColorInfo()
        
        gatingFrame = ttk.Frame(self.regexBox)
        gatingFrame.pack(anchor="w", pady=(3,0))
        self.regexGateChk = tk.BooleanVar(value=False)
        self.regexGateCheck = ttk.Checkbutton(gatingFrame, text="Apply gating", variable=self.regexGateChk, command=self._updRegexPlot)
        self.regexGateCheck.pack(side=tk.LEFT, padx=3)
        self.regexPhaseCheck = ttk.Checkbutton(
            gatingFrame,
            text="Show phase",
            variable=self.regexPhaseChk,
            command=self._updRegexPlot
        )
        self.regexPhaseCheck.pack(side=tk.LEFT, padx=3)
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
        
        self.trainBox = ttk.Frame(lfrm)
        
        if self.modelTrainingAvailable:
            ttk.Label(self.trainBox, text="Training Data Directories:").pack(anchor="w", pady=(5,0))
            dirFrame = ttk.Frame(self.trainBox)
            dirFrame.pack(anchor="w", fill=tk.X, pady=(2,0))
            self.trainDirListbox = tk.Listbox(dirFrame, height=4, width=40)
            self.trainDirListbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            dirBtnFrame = ttk.Frame(dirFrame)
            dirBtnFrame.pack(side=tk.LEFT, padx=(5,0))
            ttk.Button(dirBtnFrame, text="Add", command=self._addTrainDir, width=8).pack(pady=(0,2))
            ttk.Button(dirBtnFrame, text="Remove", command=self._removeTrainDir, width=8).pack(pady=(0,2))
            ttk.Button(dirBtnFrame, text="Clear", command=self._clearTrainDirs, width=8).pack()
            
            ttk.Label(self.trainBox, text="Model Configuration:", font=("", 9, "bold")).pack(anchor="w", pady=(10,5))
            
            archFrame = ttk.LabelFrame(self.trainBox, text="Architecture", padding=5)
            archFrame.pack(anchor="w", fill=tk.X, pady=(0,5))
            
            row1 = ttk.Frame(archFrame)
            row1.pack(anchor="w", pady=2)
            ttk.Label(row1, text="Hidden dim:").pack(side=tk.LEFT)
            ttk.Entry(row1, textvariable=self.trainHiddenDim, width=8).pack(side=tk.LEFT, padx=(5,15))
            ttk.Label(row1, text="Latent dim:").pack(side=tk.LEFT)
            ttk.Entry(row1, textvariable=self.trainLatentDim, width=8).pack(side=tk.LEFT, padx=(5,0))
            
            row2 = ttk.Frame(archFrame)
            row2.pack(anchor="w", pady=2)
            ttk.Label(row2, text="Dropout:").pack(side=tk.LEFT)
            ttk.Entry(row2, textvariable=self.trainDropout, width=8).pack(side=tk.LEFT, padx=(5,15))
            ttk.Checkbutton(row2, text="Batch norm", variable=self.trainBatchNorm).pack(side=tk.LEFT)
            
            trainFrame = ttk.LabelFrame(self.trainBox, text="Training", padding=5)
            trainFrame.pack(anchor="w", fill=tk.X, pady=(0,5))
            
            row3 = ttk.Frame(trainFrame)
            row3.pack(anchor="w", pady=2)
            ttk.Label(row3, text="Learning rate:").pack(side=tk.LEFT)
            lr_entry = ttk.Entry(row3, textvariable=self.trainLR, width=10)
            lr_entry.pack(side=tk.LEFT, padx=(5,15))
            ttk.Label(row3, text="Weight decay:").pack(side=tk.LEFT)
            wd_entry = ttk.Entry(row3, textvariable=self.trainWeightDecay, width=10)
            wd_entry.pack(side=tk.LEFT, padx=(5,0))
            
            row4 = ttk.Frame(trainFrame)
            row4.pack(anchor="w", pady=2)
            ttk.Label(row4, text="Epochs:").pack(side=tk.LEFT)
            ttk.Entry(row4, textvariable=self.trainEpochs, width=8).pack(side=tk.LEFT, padx=(5,15))
            ttk.Label(row4, text="Patience:").pack(side=tk.LEFT)
            ttk.Entry(row4, textvariable=self.trainPatience, width=8).pack(side=tk.LEFT, padx=(5,0))
            
            row5 = ttk.Frame(trainFrame)
            row5.pack(anchor="w", pady=2)
            ttk.Label(row5, text="Loss:").pack(side=tk.LEFT)
            lossCombo = ttk.Combobox(row5, textvariable=self.trainLossFn, width=10, 
                                    values=["mse", "mae", "huber", "smooth_l1"], state="readonly")
            lossCombo.pack(side=tk.LEFT, padx=(5,15))
            ttk.Label(row5, text="Device:").pack(side=tk.LEFT)
            deviceCombo = ttk.Combobox(row5, textvariable=self.trainDevice, width=8,
                                      values=["cpu", "cuda"], state="readonly")
            deviceCombo.pack(side=tk.LEFT, padx=(5,0))
            
            augFrame = ttk.LabelFrame(self.trainBox, text="Augmentation", padding=5)
            augFrame.pack(anchor="w", fill=tk.X, pady=(0,5))
            
            augRow = ttk.Frame(augFrame)
            augRow.pack(anchor="w")
            ttk.Checkbutton(augRow, text="Enable augmentation", variable=self.trainAugment).pack(side=tk.LEFT)
            ttk.Label(augRow, text="Noise std:").pack(side=tk.LEFT, padx=(15,5))
            ttk.Entry(augRow, textvariable=self.trainNoise, width=8).pack(side=tk.LEFT)
            
            actionFrame = ttk.Frame(self.trainBox)
            actionFrame.pack(anchor="w", pady=(10,0))
            ttk.Button(actionFrame, text="Run Cross-Validation", command=self._runCrossValidation).pack(side=tk.LEFT, padx=(0,5))
            ttk.Button(actionFrame, text="Train Final Model", command=self._trainFinalModel).pack(side=tk.LEFT, padx=(0,5))
            ttk.Button(actionFrame, text="Load Model", command=self._loadModel).pack(side=tk.LEFT, padx=(0,5))
            ttk.Button(actionFrame, text="Scan Files", command=self._scanTrainingFiles).pack(side=tk.LEFT)
            
            self.trainProgress = ttk.Progressbar(self.trainBox, mode='indeterminate')
            self.trainProgress.pack(fill=tk.X, pady=(10,0))
            
        else:
            msgFrame = ttk.Frame(self.trainBox)
            msgFrame.pack(expand=True, fill=tk.BOTH, pady=50)
            ttk.Label(msgFrame, text="Model Training Not Available", 
                     font=("", 12, "bold")).pack()
            ttk.Label(msgFrame, text="water_pred.py module not found or PyTorch not installed", 
                     font=("", 10)).pack(pady=(10,0))
            ttk.Label(msgFrame, text="To enable model training:", 
                     font=("", 9)).pack(pady=(20,5))
            ttk.Label(msgFrame, text="1. Ensure water_pred.py is in the same directory", 
                     font=("", 9)).pack()
            ttk.Label(msgFrame, text="2. Install PyTorch: pip install torch", 
                     font=("", 9)).pack()
            ttk.Label(msgFrame, text="3. Install other dependencies: scikit-learn, matplotlib", 
                     font=("", 9)).pack()

        self.shapeBox = ttk.Frame(lfrm)
        ttk.Label(self.shapeBox, text="S-parameter:").pack(anchor="w", pady=(5,0))
        sp = ttk.Frame(self.shapeBox); sp.pack(anchor="w", pady=(2,0))
        self.shapeParam = tk.StringVar(value="s21")
        for n in ("s11","s12","s21","s22"):
            ttk.Radiobutton(sp, text=n.upper(), value=n, variable=self.shapeParam, command=lambda:self._updShapePlot()).pack(side=tk.LEFT, padx=2)
        self.shapeNormalize = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.shapeBox, text="Normalize", variable=self.shapeNormalize, command=self._updShapePlot).pack(anchor="w", pady=(5,0))
        mt = ttk.Frame(self.shapeBox); mt.pack(anchor="w", pady=(5,0))
        self.shapeMetric = tk.StringVar(value="xcorr")
        for t,v in (("Cross-corr","xcorr"),("L1","l1"),("L2","l2"),("Adaptive L1","al1"),("Adaptive L2","al2")):
            ttk.Radiobutton(mt, text=t, value=v, variable=self.shapeMetric, command=self._updShapePlot).pack(side=tk.LEFT, padx=4)

        adaptFrame = ttk.LabelFrame(self.shapeBox, text="Adaptive Parameters")
        adaptFrame.pack(anchor="w", pady=(5,0), fill="x")
        p1 = ttk.Frame(adaptFrame); p1.pack(anchor="w")
        ttk.Label(p1, text="α:").pack(side=tk.LEFT)
        ttk.Spinbox(p1, from_=1.0, to=4.0, increment=0.5, textvariable=self.shapeAlpha, width=6, command=self._updShapePlot).pack(side=tk.LEFT, padx=(2,10))
        ttk.Label(p1, text="σ:").pack(side=tk.LEFT)
        ttk.Spinbox(p1, from_=2, to=20, textvariable=self.shapeSigma, width=6, command=self._updShapePlot).pack(side=tk.LEFT, padx=(2,10))
        ttk.Label(p1, text="γ:").pack(side=tk.LEFT)
        ttk.Spinbox(p1, from_=0.5, to=4.0, increment=0.5, textvariable=self.shapeGamma, width=6, command=self._updShapePlot).pack(side=tk.LEFT, padx=(2,10))
        ttk.Label(p1, text="w_min:").pack(side=tk.LEFT)
        ttk.Spinbox(p1, from_=0.0, to=1.0, increment=0.1, textvariable=self.shapeWmin, width=6, command=self._updShapePlot).pack(side=tk.LEFT, padx=(2,0))
        p2 = ttk.Frame(adaptFrame); p2.pack(anchor="w", pady=(3,3))
        ttk.Radiobutton(p2, text="Bidirectional", value="bidirectional", variable=self.shapeActivityType, command=self._updShapePlot).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(p2, text="Non-increasing only", value="nonincreasing", variable=self.shapeActivityType, command=self._updShapePlot).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(p2, text="Show activity", variable=self.shapeShowActivity, command=self._updShapePlot).pack(side=tk.LEFT, padx=(20,2))

        lagFrame = ttk.Frame(self.shapeBox)
        lagFrame.pack(anchor="w", pady=(5,0))
        ttk.Label(lagFrame, text="Max lag (samples):").pack(side=tk.LEFT)
        lagSpin = ttk.Spinbox(lagFrame, from_=1, to=1000, textvariable=self.maxLag, width=8, command=self._updShapePlot)
        lagSpin.pack(side=tk.LEFT, padx=(5,0))
        ttk.Label(lagFrame, text="(0 = no limit)", font=("", 8), foreground="gray").pack(side=tk.LEFT, padx=(5,0))

        btnFrame = ttk.Frame(self.shapeBox); btnFrame.pack(anchor="w", pady=(5,0))
        ttk.Button(btnFrame, text="Export matrix", command=self._exportShapeMatrix).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(btnFrame, text="Export as CSV", command=self._exportShapeCSV).pack(side=tk.LEFT)
        self.shapeData = None

        lagFrame = ttk.Frame(self.shapeBox)
        lagFrame.pack(anchor="w", pady=(5,0))
        ttk.Label(lagFrame, text="Max lag (samples):").pack(side=tk.LEFT)
        lagSpin = ttk.Spinbox(lagFrame, from_=1, to=1000, textvariable=self.maxLag, 
                              width=8, command=self._updShapePlot)
        lagSpin.pack(side=tk.LEFT, padx=(5,0))
        ttk.Label(lagFrame, text="(0 = no limit)", font=("", 8), foreground="gray").pack(side=tk.LEFT, padx=(5,0))

        self.integBox = ttk.Frame(lfrm)
        ttk.Label(self.integBox, text="S-parameters to integrate:").pack(anchor="w", pady=(5,0))
        sparamFrame = ttk.Frame(self.integBox)
        sparamFrame.pack(anchor="w", pady=(2,0))
        self.integS11 = tk.BooleanVar(value=True)
        self.integS12 = tk.BooleanVar(value=False)
        self.integS21 = tk.BooleanVar(value=True)
        self.integS22 = tk.BooleanVar(value=False)
        ttk.Checkbutton(sparamFrame, text="S11", variable=self.integS11, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(sparamFrame, text="S12", variable=self.integS12, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(sparamFrame, text="S21", variable=self.integS21, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(sparamFrame, text="S22", variable=self.integS22, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)

        ttk.Label(self.integBox, text="Integration scale:").pack(anchor="w", pady=(5,0))
        scaleFrame = ttk.Frame(self.integBox)
        scaleFrame.pack(anchor="w", pady=(2,0))
        self.integScale = tk.StringVar(value="db")
        ttk.Radiobutton(scaleFrame, text="dB scale", value="db", variable=self.integScale, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(scaleFrame, text="Linear scale", value="linear", variable=self.integScale, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)

        btnFrame = ttk.Frame(self.integBox)
        btnFrame.pack(anchor="w", pady=(5,0))
        ttk.Button(btnFrame, text="Export results", command=self._exportIntegResults).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(btnFrame, text="Export as CSV", command=self._exportIntegCSV).pack(side=tk.LEFT)
        ttk.Label(self.integBox, text="Sort by:").pack(anchor="w", pady=(5,0))
        sortFrame = ttk.Frame(self.integBox)
        sortFrame.pack(anchor="w", pady=(2,0))
        self.integSortBy = tk.StringVar(value="name")
        ttk.Radiobutton(sortFrame, text="File name", value="name", variable=self.integSortBy, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sortFrame, text="S11", value="s11", variable=self.integSortBy, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sortFrame, text="S12", value="s12", variable=self.integSortBy, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sortFrame, text="S21", value="s21", variable=self.integSortBy, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sortFrame, text="S22", value="s22", variable=self.integSortBy, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sortFrame, text="Total", value="total", variable=self.integSortBy, command=self._updIntegPlot).pack(side=tk.LEFT, padx=2)
        self.integSortAsc = tk.BooleanVar(value=True)
        ttk.Checkbutton(sortFrame, text="Ascending", variable=self.integSortAsc, command=self._updIntegPlot).pack(side=tk.LEFT, padx=(10,2))

        self.integData = None


        self.tdAnalysisBox = ttk.Frame(lfrm)
        ttk.Label(self.tdAnalysisBox, text="Time Domain Analysis", font=("", 10, "bold")).pack(anchor="w", pady=(5,0))
        
        ttk.Label(self.tdAnalysisBox, text="This analysis compares the first global maximum", font=("", 9)).pack(anchor="w", pady=(2,0))
        ttk.Label(self.tdAnalysisBox, text="of S11 and S21 in time domain.", font=("", 9)).pack(anchor="w")
        ttk.Label(self.tdAnalysisBox, text="Checks if: t_max(S21) > 2 * t_max(S11)", font=("", 9)).pack(anchor="w", pady=(5,0))
        
        btnFrame = ttk.Frame(self.tdAnalysisBox)
        btnFrame.pack(anchor="w", pady=(10,0))
        ttk.Button(btnFrame, text="Export Analysis", command=self._exportTDAnalysis).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(btnFrame, text="Clear Markers", command=self._clearTDAnalysisMarkers).pack(side=tk.LEFT)
        
        optFrame = ttk.Frame(self.tdAnalysisBox)
        optFrame.pack(anchor="w", pady=(5,0))
        self.tdAnalysisTimeLimit = ValidatedDoubleVar(value=50.0)
        ttk.Label(optFrame, text="Time limit [ns]:").pack(side=tk.LEFT)
        timeLimitEntry = ttk.Entry(optFrame, textvariable=self.tdAnalysisTimeLimit, width=8)
        timeLimitEntry.pack(side=tk.LEFT, padx=(5,0))
        self._validateNumeric(timeLimitEntry, self.tdAnalysisTimeLimit, self._updTDAnalysisPlot)
        
        self.tdAnalysisMarkPeaks = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.tdAnalysisBox, text="Mark peaks on plot", variable=self.tdAnalysisMarkPeaks, 
                       command=self._updTDAnalysisPlot).pack(anchor="w", pady=(5,0))

        nb = ttk.Notebook(self)
        nb.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.nb = nb
        
        frmF = ttk.Frame(nb)
        nb.add(frmF, text="Wykresy")
        self.nb.bind("<<NotebookTabChanged>>", self._onTab)
        
        self.plotPane = ttk.PanedWindow(frmF, orient=tk.HORIZONTAL)
        self.plotPane.pack(fill=tk.BOTH, expand=True)
        
        plotFrame = ttk.Frame(self.plotPane)
        self.plotPane.add(plotFrame, weight=3)
        
        self.fig, self.ax = plt.subplots(figsize=(10,7))
        self.cv = FigureCanvasTkAgg(self.fig, master=plotFrame)
        canvas_widget = self.cv.get_tk_widget()
        
        toolbar_frame = ttk.Frame(plotFrame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tb = NavigationToolbar2Tk(self.cv, toolbar_frame)
        self.tb.update()
        self.tb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.legendFrame = ttk.Frame(self.plotPane)
        self.plotPane.add(self.legendFrame, weight=1)
        
        ttk.Label(self.legendFrame, text="Legend", font=("", 10, "bold")).pack(pady=5)
        
        legendScroll = ttk.Scrollbar(self.legendFrame)
        legendScroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.legendCanvas = tk.Canvas(self.legendFrame, yscrollcommand=legendScroll.set, width=150)
        self.legendCanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        legendScroll.config(command=self.legendCanvas.yview)
        
        self.legendItemsFrame = ttk.Frame(self.legendCanvas)
        self.legendCanvas.create_window((0,0), window=self.legendItemsFrame, anchor="nw")
        
        def updateLegendScroll(event=None):
            self.legendCanvas.configure(scrollregion=self.legendCanvas.bbox("all"))
        self.legendItemsFrame.bind("<Configure>", updateLegendScroll)
        
        frmT = ttk.Frame(nb)
        nb.add(frmT, text="Time domain")
        
        self.plotPaneT = ttk.PanedWindow(frmT, orient=tk.HORIZONTAL)
        self.plotPaneT.pack(fill=tk.BOTH, expand=True)
        
        plotFrameT = ttk.Frame(self.plotPaneT)
        self.plotPaneT.add(plotFrameT, weight=3)
        
        self.figT, self.axT = plt.subplots(figsize=(10,10))
        self.cvT = FigureCanvasTkAgg(self.figT, master=plotFrameT)
        canvas_widget = self.cvT.get_tk_widget()
        
        toolbar_frameT = ttk.Frame(plotFrameT)
        toolbar_frameT.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tbT = NavigationToolbar2Tk(self.cvT, toolbar_frameT)
        self.tbT.update()
        self.tbT.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.legendFrameT = ttk.Frame(self.plotPaneT)
        self.plotPaneT.add(self.legendFrameT, weight=1)
        
        ttk.Label(self.legendFrameT, text="Legend", font=("", 10, "bold")).pack(pady=5)
        
        legendScrollT = ttk.Scrollbar(self.legendFrameT)
        legendScrollT.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.legendCanvasT = tk.Canvas(self.legendFrameT, yscrollcommand=legendScrollT.set, width=150)
        self.legendCanvasT.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        legendScrollT.config(command=self.legendCanvasT.yview)
        
        self.legendItemsFrameT = ttk.Frame(self.legendCanvasT)
        self.legendCanvasT.create_window((0,0), window=self.legendItemsFrameT, anchor="nw")
        
        def updateLegendScrollT(event=None):
            self.legendCanvasT.configure(scrollregion=self.legendCanvasT.bbox("all"))
        self.legendItemsFrameT.bind("<Configure>", updateLegendScrollT)
        
        frmR = ttk.Frame(nb)
        nb.add(frmR, text="Regex Highlighting")
        
        self.plotPaneR = ttk.PanedWindow(frmR, orient=tk.HORIZONTAL)
        self.plotPaneR.pack(fill=tk.BOTH, expand=True)
        
        plotFrameR = ttk.Frame(self.plotPaneR)
        self.plotPaneR.add(plotFrameR, weight=3)
        
        self.figR, self.axR = plt.subplots(figsize=(10,7))
        self.cvR = FigureCanvasTkAgg(self.figR, master=plotFrameR)
        self.cvR.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frameR = ttk.Frame(plotFrameR)
        toolbar_frameR.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tbR = NavigationToolbar2Tk(self.cvR, toolbar_frameR)
        self.tbR.update()
        self.tbR.pack(side=tk.LEFT, fill=tk.X, expand=True)

        
        self.legendFrameR = ttk.Frame(self.plotPaneR)
        self.plotPaneR.add(self.legendFrameR, weight=1)
        
        ttk.Label(self.legendFrameR, text="Legend", font=("", 10, "bold")).pack(pady=5)
        
        legendScrollR = ttk.Scrollbar(self.legendFrameR)
        legendScrollR.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.legendCanvasR = tk.Canvas(self.legendFrameR, yscrollcommand=legendScrollR.set, width=150)
        self.legendCanvasR.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        legendScrollR.config(command=self.legendCanvasR.yview)
        
        self.legendItemsFrameR = ttk.Frame(self.legendCanvasR)
        self.legendCanvasR.create_window((0,0), window=self.legendItemsFrameR, anchor="nw")
        
        def updateLegendScrollR(event=None):
            self.legendCanvasR.configure(scrollregion=self.legendCanvasR.bbox("all"))
        self.legendItemsFrameR.bind("<Configure>", updateLegendScrollR)
        
        if not self.legendVisible.get():
            self.plotPane.forget(self.legendFrame)
            self.plotPaneT.forget(self.legendFrameT)
            self.plotPaneR.forget(self.legendFrameR)
        
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
        self.figV, self.axV = plt.subplots(figsize=(10,8))
        self.cvV = FigureCanvasTkAgg(self.figV, master=frmV)
        self.cvV.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tbV = NavigationToolbar2Tk(self.cvV, frmV)
        self.tbV.update()
        self.tbV.pack(fill=tk.X)
        
        frmM = ttk.Frame(nb)
        nb.add(frmM, text="Model Training")
        if self.modelTrainingAvailable:
            self.figM, (self.axM1, self.axM2) = plt.subplots(1, 2, figsize=(12,5))
            self.cvM = FigureCanvasTkAgg(self.figM, master=frmM)
            self.cvM.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.tbM = NavigationToolbar2Tk(self.cvM, frmM)
            self.tbM.update()
            self.tbM.pack(fill=tk.X)
            self.axM1.text(0.5, 0.5, "No training results yet", ha="center", va="center", fontsize=12, color="gray")
            self.axM1.set_xticks([])
            self.axM1.set_yticks([])
            self.axM2.text(0.5, 0.5, "Run cross-validation or train a model", ha="center", va="center", fontsize=12, color="gray")
            self.axM2.set_xticks([])
            self.axM2.set_yticks([])
            self.figM.tight_layout()
            self.cvM.draw()
        else:
            lblFrame = ttk.Frame(frmM)
            lblFrame.pack(expand=True)
            ttk.Label(lblFrame, text="Model training not available", font=("", 14)).pack()

        frmSC = ttk.Frame(nb)
        nb.add(frmSC, text="Shape Comparison")
        self.figSC, self.axSC = plt.subplots(figsize=(10,8))
        self.cvSC = FigureCanvasTkAgg(self.figSC, master=frmSC)
        self.cvSC.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tbSC = NavigationToolbar2Tk(self.cvSC, frmSC)
        self.tbSC.update()
        self.tbSC.pack(fill=tk.X)

        frmI = ttk.Frame(nb)
        nb.add(frmI, text="Integration")
        self.figI, self.axI = plt.subplots(figsize=(10,8))
        self.cvI = FigureCanvasTkAgg(self.figI, master=frmI)
        self.cvI.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tbI = NavigationToolbar2Tk(self.cvI, frmI)
        self.tbI.update()
        self.tbI.pack(fill=tk.X)

        self.cv.mpl_connect('button_press_event', self._onClick)
        self.cvT.mpl_connect('button_press_event', self._onClick)
        self.cv.mpl_connect('pick_event', self._onPick)
        self.cvT.mpl_connect('pick_event', self._onPick)
        
        self.cvV.mpl_connect('button_press_event', self._onClickVar)
        self.cvV.mpl_connect('pick_event', self._onPickVar)
        self.cvV.mpl_connect('motion_notify_event', self._onMotionVar)

        frmTDA = ttk.Frame(nb)
        nb.add(frmTDA, text="TD Analysis")
        self.figTDA, (self.axTDA1, self.axTDA2) = plt.subplots(2, 1, figsize=(10,8), sharex=True)
        self.cvTDA = FigureCanvasTkAgg(self.figTDA, master=frmTDA)
        self.cvTDA.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tbTDA = NavigationToolbar2Tk(self.cvTDA, frmTDA)
        self.tbTDA.update()
        self.tbTDA.pack(fill=tk.X)
        
        self.cvTDA.mpl_connect('button_press_event', self._onClickTDAnalysis)

    def _updateColorInfo(self):
        if self.regexSmallDiffChk.get():
            threshold = self.regexSmallDiffThreshold.get()
            ref_db = self.regexSmallDiffRefDB.get()
            min_count = self.regexSmallDiffMinCount.get()
            if min_count == 1:
                text = f"Green: monotonic | Blue: any difference <= {threshold} @ {ref_db}dB scaled"
            else:
                text = f"Green: monotonic | Blue: >={min_count} differences <= {threshold} @ {ref_db}dB scaled"
        else:
            text = "Green: monotonic regions"
        self.colorInfoLabel.config(text=text)


    def _avgFiles(self):
        dialog = tk.Toplevel(self)
        dialog.title("Average S-parameter files")
        dialog.geometry("400x500")
        dialog.transient(self)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select averaging mode:", font=("", 10, "bold")).pack(pady=10)
        
        mode = tk.StringVar(value="loaded")
        ttk.Radiobutton(dialog, text="Average loaded files", variable=mode, value="loaded").pack(anchor="w", padx=20)
        ttk.Radiobutton(dialog, text="Load and average new files", variable=mode, value="new").pack(anchor="w", padx=20)
        
        ttk.Separator(dialog, orient="horizontal").pack(fill="x", pady=10)
        
        loadedFrame = ttk.LabelFrame(dialog, text="Select loaded files to average")
        loadedFrame.pack(fill="both", expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(loadedFrame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(loadedFrame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.config(command=listbox.yview)
        
        file_indices = []
        for i, (v, p, d) in enumerate(self.fls):
            if not d.get('is_average', False):
                listbox.insert(tk.END, Path(p).name)
                file_indices.append(i)
        
        nameFrame = ttk.Frame(dialog)
        nameFrame.pack(fill="x", padx=10, pady=5)
        ttk.Label(nameFrame, text="Average name:").pack(side=tk.LEFT)
        nameVar = tk.StringVar(value=f"average_{self.avgCounter+1}")
        ttk.Entry(nameFrame, textvariable=nameVar, width=30).pack(side=tk.LEFT, padx=(5,0))
        
        def process():
            if mode.get() == "loaded":
                selected = listbox.curselection()
                if len(selected) < 2:
                    messagebox.showwarning("Selection error", "Select at least 2 files to average")
                    return
                
                indices = [file_indices[s] for s in selected]
                self._performAverage(indices, nameVar.get())
                dialog.destroy()
            else:
                dialog.destroy()
                self._loadAndAverage(nameVar.get())
        
        buttonFrame = ttk.Frame(dialog)
        buttonFrame.pack(pady=10)
        ttk.Button(buttonFrame, text="OK", command=process).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttonFrame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")


        
    def _compute_derivatives(self, signals):
        n_signals, n_points = signals.shape
        derivs = np.zeros_like(signals)
        for i in range(n_signals):
            derivs[i, 1:-1] = (signals[i, 2:] - signals[i, :-2]) / 2
            derivs[i, 0] = signals[i, 1] - signals[i, 0]
            derivs[i, -1] = signals[i, -1] - signals[i, -2]
        return derivs

    def _compute_activity_field(self, derivs, alpha, activity_type):
        if activity_type == "nonincreasing":
            activity_contrib = np.where(derivs <= 0, np.abs(derivs)**alpha, 0)
        else:
            activity_contrib = np.abs(derivs)**alpha
            # activity_contrib = np.where(derivs >= 0, np.abs(derivs)**alpha, 0)
        return np.mean(activity_contrib, axis=0)

    def _gaussian_smooth(self, field, sigma):
        if sigma == 0:
            return field
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(field, sigma, mode='reflect')

    def _compute_adaptive_weights(self, activity, gamma, w_min):
        eps = 1e-10
        activity_max = np.max(activity) + eps
        weights = w_min + (1 - w_min) * (activity / activity_max)**gamma
        weights *= len(weights) / np.sum(weights)
        return weights

    def _adaptive_distance(self, sig1, sig2, weights, p):
        diff = np.abs(sig1 - sig2)**p
        return np.sum(weights * diff)**(1/p)
        
    def _performAverage(self, indices, name):
        fmin = self.fmin.get()
        fmax = self.fmax.get()
        sstr = f"{fmin}-{fmax}ghz"
        
        networks = []
        for idx in indices:
            v, p, d = self.fls[idx]
            ext = Path(p).suffix.lower()
            if ext not in ['.s1p', '.s2p', '.s3p']:
                continue
            
            try:
                ntw_full = d.get('ntwk_full')
                if ntw_full is None:
                    ntw_full = loadFile(p)
                    d['ntwk_full'] = ntw_full
                
                networks.append(ntw_full)
                v.set(False)
            except:
                pass
        
        if len(networks) < 2:
            messagebox.showwarning("Error", "Could not load enough valid S-parameter files")
            return
        
        avg_network = self._computeAverage(networks)
        
        if avg_network is None:
            messagebox.showwarning("Error", "Failed to compute average")
            return
        
        self.avgCounter += 1
        v = tk.BooleanVar(value=True)
        chk = tk.Checkbutton(self.fbox, text=f"[AVG] {name}", variable=v, command=self._updAll, fg="blue", activeforeground="blue")
        chk.pack(anchor="w")
        
        avg_path = f"<average_{self.avgCounter}>"
        d = {
            'ntwk_full': avg_network,
            'is_average': True,
            'source_files': [self.fls[i][1] for i in indices],
            'line_color': None,
            'line_width': 2.0,
            'custom_name': name
        }
        
        chk.bind("<Button-3>", lambda e: self._showStyleMenu(e, avg_path))
        self.fls.append((v, avg_path, d))
        
        self.fileListCanvas.configure(scrollregion=self.fileListCanvas.bbox("all"))
        self._updAll()
    
    def _loadAndAverage(self, name):
        files = filedialog.askopenfilenames(
            title="Select S-parameter files to average",
            filetypes=[("S-parameter files", "*.s1p *.s2p *.s3p"), ("All files", "*.*")]
        )
        
        if len(files) < 2:
            return
        
        networks = []
        for filepath in files:
            try:
                ntw = loadFile(filepath)
                networks.append(ntw)
            except:
                pass
        
        if len(networks) < 2:
            messagebox.showwarning("Error", "Could not load enough valid S-parameter files")
            return
        
        avg_network = self._computeAverage(networks)
        
        if avg_network is None:
            messagebox.showwarning("Error", "Failed to compute average")
            return
        
        self.avgCounter += 1
        v = tk.BooleanVar(value=True)
        chk = tk.Checkbutton(self.fbox, text=f"[AVG] {name}", variable=v, command=self._updAll, fg="blue", activeforeground="blue")
        chk.pack(anchor="w")
        
        avg_path = f"<average_{self.avgCounter}>"
        d = {
            'ntwk_full': avg_network,
            'is_average': True,
            'source_files': files,
            'line_color': None,
            'line_width': 2.0,
            'custom_name': name
        }
        
        chk.bind("<Button-3>", lambda e: self._showStyleMenu(e, avg_path))
        self.fls.append((v, avg_path, d))
        
        self.fileListCanvas.configure(scrollregion=self.fileListCanvas.bbox("all"))
        self._updAll()
    
    def _computeAverage(self, networks):
        if not networks:
            return None
        
        try:
            ref_freqs = networks[0].f
            n_ports = networks[0].nports
            
            for ntw in networks[1:]:
                if ntw.nports != n_ports:
                    messagebox.showwarning("Error", "All networks must have same number of ports")
                    return None
                if len(ntw.f) != len(ref_freqs) or not np.allclose(ntw.f, ref_freqs, rtol=1e-6):
                    messagebox.showwarning("Error", "All networks must have same frequency points")
                    return None
            
            avg_s = np.zeros_like(networks[0].s, dtype=complex)
            for ntw in networks:
                avg_s += ntw.s
            avg_s /= len(networks)
            
            avg_network = rf.Network()
            avg_network.s = avg_s
            avg_network.frequency = networks[0].frequency
            avg_network.z0 = networks[0].z0
            
            return avg_network
            
        except Exception as e:
            return None


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
            return [], None, None, [], []
            
        files_data.sort(key=lambda x: x[1])
        labels = [d[0] for d in files_data]
        freqs = files_data[0][2]
        s_matrix = np.array([d[3] for d in files_data])
        
        ordered_indices = []
        strict_monotonic = self.regexStrictMonotonic.get()
        threshold = self.regexSmallDiffThreshold.get()
        
        for i in range(s_matrix.shape[1]):
            values = s_matrix[:, i]
            diffs = np.diff(values)
            
            is_monotonic = False
            if strict_monotonic:
                if np.all(diffs > 0) or np.all(diffs < 0):
                    is_monotonic = True
            else:
                if np.all(diffs >= 0) or np.all(diffs <= 0):
                    is_monotonic = True
            
            if is_monotonic:
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
        
        small_diff_ranges = []
        if self.regexSmallDiffChk.get() and ordered_indices:
            min_count = self.regexSmallDiffMinCount.get()
            base_threshold = self.regexSmallDiffThreshold.get()
            ref_db = self.regexSmallDiffRefDB.get()
            
            for i in ordered_indices:
                values = s_matrix[:, i]
                diffs = np.diff(values)
                abs_diffs = np.abs(diffs)
                
                avg_level = np.mean(values)
                if ref_db != 0:
                    scaled_threshold = base_threshold * abs(avg_level) / abs(ref_db)
                else:
                    scaled_threshold = base_threshold
                
                scaled_threshold = max(scaled_threshold, base_threshold * 0.1)
                
                small_diff_count = np.sum(abs_diffs <= scaled_threshold)
                
                if small_diff_count >= min_count:
                    small_diff_ranges.append((freqs[i], freqs[i]))
        
        merged_small_diff_ranges = []
        if small_diff_ranges:
            start_freq = small_diff_ranges[0][0]
            end_freq = small_diff_ranges[0][1]
            
            for i in range(1, len(small_diff_ranges)):
                curr_start, curr_end = small_diff_ranges[i]
                freq_idx_prev = np.where(freqs == end_freq)[0][0]
                freq_idx_curr = np.where(freqs == curr_start)[0][0]
                
                if freq_idx_curr == freq_idx_prev + 1:
                    end_freq = curr_end
                else:
                    merged_small_diff_ranges.append((start_freq, end_freq))
                    start_freq = curr_start
                    end_freq = curr_end
            
            merged_small_diff_ranges.append((start_freq, end_freq))
        
        return freq_ranges, freqs, s_matrix, labels, merged_small_diff_ranges

    def _updRegexPlot(self):
        self.figR.clear()

        pattern = self.regexPattern.get()
        param = self.regexParam.get()
        fmin = self.fmin.get()
        fmax = self.fmax.get()
        sstr = f"{fmin}-{fmax}ghz"

        self.regexSpans = []
        files_data = []
        all_files_info = []

        for v, p, d in self.fls:
            if d.get('is_average', False) and d.get('custom_name'):
                fname = d.get('custom_name')
            else:
                fname = Path(p).stem
            value = self._extractValue(fname, pattern)

            if v.get():
                all_files_info.append(
                    f"{'Y' if value is not None else 'N'} {fname} -> "
                    f"{value if value is not None else 'no match'}"
                )

            if not v.get() or value is None:
                continue

            ext = Path(p).suffix.lower()
            try:
                if d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']:
                    ntw_full = d.get('ntwk_full')
                    if ntw_full is None:
                        ntw_full = loadFile(p)
                        d['ntwk_full'] = ntw_full

                    ntw = ntw_full[sstr]

                    if self.regexGateChk.get():
                        center = self.gateCenter.get()
                        span   = self.gateSpan.get()
                        s_param = getattr(ntw, param)
                        s_gated = s_param.time_gate(center=center, span=span)
                        freq = ntw.f
                        raw_data = s_gated
                    else:
                        freq = ntw.f
                        raw_data = getattr(ntw, param)

                    if self.regexPhaseChk.get():
                        phase_rad  = np.unwrap(np.angle(raw_data.s.flatten()))
                        phase_deg  = np.degrees(phase_rad)
                        s_data     = (phase_deg + 180) % 360 - 180 

                    else:
                        s_data    = raw_data.s_db.flatten()

                    files_data.append((fname, value, freq, s_data, d))

            except Exception:
                pass

        self.txt.config(state=tk.NORMAL)
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, f"Regex: {pattern} (group {self.regexGroup.get()})\n")
        self.txt.insert(tk.END, f"Frequency range: {fmin}-{fmax} GHz\n")
        if self.regexGateChk.get():
            self.txt.insert(
                tk.END,
                f"Gating: {self.gateCenter.get()} +/- {self.gateSpan.get()/2} ns\n"
            )
        if self.regexHighlightChk.get():
            mode = "Strict monotonic" if self.regexStrictMonotonic.get() else "Non-strict monotonic"
            self.txt.insert(tk.END, f"Highlighting: {mode}\n")
        if self.regexSmallDiffChk.get():
            self.txt.insert(
                tk.END,
                f"Small diff threshold: {self.regexSmallDiffThreshold.get()} @ {self.regexSmallDiffRefDB.get()} dB (scaled)\n"
            )
        self.txt.insert(tk.END, "-" * 40 + "\n")
        for info in all_files_info:
            self.txt.insert(tk.END, info + "\n")
        self.txt.config(state=tk.DISABLED)

        ax = self.figR.add_subplot(111)

        if not files_data:
            ax.text(
                0.5, 0.5,
                f"No files match pattern: {pattern}\nCheck regex and capture group",
                ha="center", va="center", fontsize=12, color="gray"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self.cvR.draw()
            return

        analysis_data = [(f, v, fr, sd) for f, v, fr, sd, _ in files_data]
        ranges, freqs, s_matrix, labels, small_diff_ranges = self._analyzeOrderedRanges(analysis_data)
        self.regexRanges = ranges

        files_data.sort(key=lambda x: x[1])

        legend_items = []
        for i, (fname, value, freq, s_data, file_dict) in enumerate(files_data):
            kwargs = {'label': fname}
            if file_dict.get('line_color'):
                kwargs['color'] = file_dict['line_color']
            else:
                cmap = plt.cm.viridis(np.linspace(0, 1, len(files_data)))
                kwargs['color'] = cmap[i]
            if file_dict.get('line_width', 1.0) != 1.0:
                kwargs['linewidth'] = file_dict['line_width']

            line, = ax.plot(freq, s_data, **kwargs)
            legend_items.append((fname, matplotlib.colors.to_hex(line.get_color())))

        if self.regexHighlightChk.get():
            for (f1, f2) in ranges:
                span = ax.axvspan(f1, f2, color='green', alpha=0.3)
                self.regexSpans.append(span)
            if self.regexSmallDiffChk.get():
                for (f1, f2) in small_diff_ranges:
                    span = ax.axvspan(f1, f2, color='blue', alpha=0.5)
                    self.regexSpans.append(span)

        for line in self.extremaLinesR:
            try:
                line.remove()
            except:
                pass
        self.extremaLinesR.clear()

        all_extrema = []

        if self.extremaEnabled.get() and files_data:
            range_min = self.extremaRangeMin.get() * 1e9
            range_max = self.extremaRangeMax.get() * 1e9
            
            for fname, value, freq, s_data, file_dict in files_data:
                mask = (freq >= range_min) & (freq <= range_max)
                if np.any(mask):
                    freq_masked = freq[mask]
                    s_data_masked = s_data[mask]
                    
                    if self.extremaMaxima.get():
                        max_idx = np.argmax(s_data_masked)
                        max_freq = freq_masked[max_idx]
                        max_val = s_data_masked[max_idx]
                        line = ax.axvline(x=max_freq, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
                        self.extremaLinesR.append(line)
                        all_extrema.append({
                            'type': 'max',
                            'freq': max_freq,
                            'value': max_val,
                            'param': param,
                            'file': fname
                        })
                    
                    if self.extremaMinima.get():
                        min_idx = np.argmin(s_data_masked)
                        min_freq = freq_masked[min_idx]
                        min_val = s_data_masked[min_idx]
                        line = ax.axvline(x=min_freq, color='blue', linestyle='--', alpha=0.3, linewidth=0.8)
                        self.extremaLinesR.append(line)
                        all_extrema.append({
                            'type': 'min',
                            'freq': min_freq,
                            'value': min_val,
                            'param': param,
                            'file': fname
                        })
            
            if all_extrema:
                self.txt.config(state=tk.NORMAL)
                self.txt.insert(tk.END, "\n" + "="*40 + "\n")
                self.txt.insert(tk.END, f"EXTREMA (Range: {self.extremaRangeMin.get():.2f}-{self.extremaRangeMax.get():.2f} GHz)\n")
                self.txt.insert(tk.END, "-"*40 + "\n")
                
                all_extrema.sort(key=lambda x: (x['type'], x['freq']))
                
                for ext in all_extrema:
                    type_str = "MAX" if ext['type'] == 'max' else "MIN"
                    freq_ghz = ext['freq'] / 1e9
                    self.txt.insert(tk.END, f"{type_str} | {ext['file']} | {ext['param'].upper()} | {freq_ghz:.4f} GHz | {ext['value']:.2f} dB\n")
                
                self.txt.config(state=tk.DISABLED)

        ax.set_xlabel("Frequency [Hz]")
        if self.regexPhaseChk.get():
            ax.set_ylabel(f"Phase {param.upper()} [°]")
        else:
            ax.set_ylabel(f"|{param.upper()}| [dB]")

        title = f"Regex-based ordering: {pattern} (group {self.regexGroup.get()})"
        if self.regexPhaseChk.get():
            title += " — Phase"
        if self.regexGateChk.get():
            title += f" — Gated ({self.gateCenter.get()}+/-{self.gateSpan.get()/2} ns)"
        ax.set_title(title)
        ax.grid(True)

        if self.legendOnPlot.get() and legend_items: 
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ncol = min(5, (len(labels) - 1) // 10 + 1)
                ax.legend(handles, labels, loc='best', ncol=ncol, fontsize=8, framealpha=0.9)

        self.figR.tight_layout()
        self.cvR.draw()
        self._updateLegendPanel(legend_items, 'regex')

    def _onTab(self, e):
        tabTxt = self.nb.tab(self.nb.select(), "text")
        if tabTxt == "Wykresy":
            self.tab = "freq"
            self.rbox.pack_forget()
            self.shapeBox.pack_forget()
            self.gateBox.pack_forget()
            self.integBox.pack_forget()
            self.regexBox.pack_forget()
            self.overlapBox.pack_forget()
            self.varBox.pack_forget()
            self.trainBox.pack_forget()
            self.cbox.pack(anchor="w", pady=(2,0))
            self.txt.config(state=tk.NORMAL)
            self.extremaBox.pack(anchor="w", pady=(2,0))
            self.txt.delete(1.0, tk.END)
            for txt in self.mtxt:
                self.txt.insert(tk.END, txt)
            self.txt.config(state=tk.DISABLED)
            self._updPlot()  
        elif tabTxt == "Time domain":
            self.tab = "time"
            self.cbox.pack_forget()
            self.regexBox.pack_forget()
            self.overlapBox.pack_forget()
            self.shapeBox.pack_forget()
            self.varBox.pack_forget()
            self.integBox.pack_forget()
            self.trainBox.pack_forget()
            self.extremaBox.pack(anchor="w", pady=(2,0))
            self.rbox.pack(anchor="w", pady=(2,0))
            self.gateBox.pack(anchor="w", pady=(8,0))
            self.txt.config(state=tk.NORMAL)
            self.txt.delete(1.0, tk.END)
            for txt in self.mtxt:
                self.txt.insert(tk.END, txt)
            self.txt.config(state=tk.DISABLED)
            self._updTPlot()  
            if os.name == 'nt' and hasattr(self, 'tbT'):
                if not self.tbT.winfo_ismapped():
                    self.tbT.pack(side=tk.BOTTOM, fill=tk.X)
                self.tbT.update()
        elif tabTxt == "Regex Highlighting":
            self.tab = "regex"
            self.cbox.pack_forget()
            self.rbox.pack_forget()
            self.extremaBox.pack(anchor="w", pady=(2,0))
            self.gateBox.pack_forget()
            self.overlapBox.pack_forget()
            self.varBox.pack_forget()
            self.trainBox.pack_forget()
            self.integBox.pack_forget()
            self.shapeBox.pack_forget()
            self.regexBox.pack(anchor="w", pady=(2,0))
            self._updRegexPlot()
        elif tabTxt == "Range Overlaps":
            self.tab = "overlap"
            self.cbox.pack_forget()
            self.rbox.pack_forget()
            self.gateBox.pack_forget()
            self.regexBox.pack_forget()
            self.integBox.pack_forget()
            self.shapeBox.pack_forget()
            self.varBox.pack_forget()
            self.trainBox.pack_forget()
            self.overlapBox.pack(anchor="w", pady=(2,0))
            self._updOverlapPlot()
        elif tabTxt == "Variance Analysis":
            self.tab = "variance"
            self.cbox.pack_forget()
            self.rbox.pack_forget()
            self.shapeBox.pack_forget()
            self.integBox.pack_forget()
            self.gateBox.pack_forget()
            self.regexBox.pack_forget()
            self.overlapBox.pack_forget()
            self.trainBox.pack_forget()
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
        elif tabTxt == "Model Training":
            self.tab = "training"
            self.cbox.pack_forget()
            self.rbox.pack_forget()
            self.gateBox.pack_forget()
            self.regexBox.pack_forget()
            self.overlapBox.pack_forget()
            self.varBox.pack_forget()
            self.integBox.pack_forget()
            self.shapeBox.pack_forget()
            self.trainBox.pack(anchor="w", pady=(2,0))
            self.txt.config(state=tk.NORMAL)
            self.txt.delete(1.0, tk.END)
            if self.modelTrainingAvailable:
                self.txt.insert(tk.END, "Model Training\n")
                self.txt.insert(tk.END, "=" * 40 + "\n")
                self.txt.insert(tk.END, "Train neural networks to predict water content\n")
                self.txt.insert(tk.END, "from S-parameter measurements.\n\n")
                self.txt.insert(tk.END, "Data directories: " + str(len(self.trainDataDirs)) + "\n")
                if self.torch and self.torch.cuda.is_available():
                    self.txt.insert(tk.END, "CUDA available: Yes\n")
                else:
                    self.txt.insert(tk.END, "CUDA available: No\n")
            else:
                self.txt.insert(tk.END, "Model training module not available\n")
            self.txt.config(state=tk.DISABLED)
        elif tabTxt == "Shape Comparison":
            self.tab = "shape"
            self.cbox.pack_forget()
            self.rbox.pack_forget()
            self.gateBox.pack_forget()
            self.regexBox.pack_forget()
            self.overlapBox.pack_forget()
            self.varBox.pack_forget()
            self.trainBox.pack_forget()
            self.integBox.pack_forget()
            self.shapeBox.pack(anchor="w", pady=(2,0))
            self._updShapePlot()
        elif tabTxt == "Integration":
            self.tab = "integration"
            self.cbox.pack_forget()
            self.rbox.pack_forget()
            self.gateBox.pack_forget()
            self.regexBox.pack_forget()
            self.overlapBox.pack_forget()
            self.varBox.pack_forget()
            self.trainBox.pack_forget()
            self.shapeBox.pack_forget()
            self.integBox.pack(anchor="w", pady=(2,0))
            self._updIntegPlot()
        elif tabTxt == "TD Analysis":
            self.tab = "td_analysis"
            self.cbox.pack_forget()
            self.rbox.pack_forget()
            self.gateBox.pack_forget()
            self.regexBox.pack_forget()
            self.overlapBox.pack_forget()
            self.varBox.pack_forget()
            self.trainBox.pack_forget()
            self.shapeBox.pack_forget()
            self.integBox.pack_forget()
            self.extremaBox.pack_forget()
            self.tdAnalysisBox.pack(anchor="w", pady=(2,0))
            self._updTDAnalysisPlot()


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
                print("Loading file:", p)
                v = tk.BooleanVar(value=True)
                chk = ttk.Checkbutton(self.fbox, text=Path(p).name, variable=v, command=self._updAll)
                chk.pack(anchor="w")
                chk.bind("<Button-3>", lambda e, path=p: self._showStyleMenu(e, path))
                d = {'line_color': None, 'line_width': 1.0}
                self.fls.append((v, p, d))
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
        
        all_legend_items = []
        seen_labels = set()
        
        for ax, prm in zip(axes, sel):
            for v, p, d in self.fls:
                if not v.get(): continue
                ext = Path(p).suffix.lower()
                if d.get('is_average', False) and d.get('custom_name'):
                    lbl = d['custom_name']
                else:
                    lbl = Path(p).stem
                try:
                    if d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']:
                        ntw_full = d.get('ntwk_full')
                        cached_range = d.get('cached_range')
                        
                        if ntw_full is None:
                            ntw_full = loadFile(p)
                            d['ntwk_full'] = ntw_full
                        
                        if cached_range != sstr:
                            ntw = ntw_full[sstr]
                            d['ntwk'] = ntw
                            d['cached_range'] = sstr
                        else:
                            ntw = d['ntwk']
                        
                        arr = getattr(ntw, prm).s_db.flatten()
                        fr = ntw.f
                    elif ext == '.csv' and prm == 's11':
                        if 'csv_data' not in d:
                            df = pd.read_csv(p)
                            d['csv_data'] = (df.iloc[:,0].values, df.iloc[:,1].values)
                        fr, arr = d['csv_data']
                        d['ntwk'] = None
                    else:
                        continue
                    
                    line_kwargs = {'label': lbl}
                    if d.get('line_color'):
                        line_kwargs['color'] = d['line_color']
                    if d.get('line_width', 1.0) != 1.0:
                        line_kwargs['linewidth'] = d['line_width']
                    
                    line = ax.plot(fr, arr, **line_kwargs)[0]
                    if lbl not in seen_labels:
                        all_legend_items.append((lbl, line.get_color()))
                        seen_labels.add(lbl)
                except Exception:
                    pass
            ax.set_ylabel(f"|{prm.upper()}| [dB]")
            ax.set_title(f"{prm.upper()} magnitude")
            ax.grid(True)

            if self.legendOnPlot.get() and prm == sel[-1]:  
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ncol = min(5, (len(labels) - 1) // 10 + 1)
                    ax.legend(handles, labels, loc='best', ncol=ncol, fontsize=8, framealpha=0.9)

        for line in self.extremaLines:
            try:
                line.remove()
            except:
                pass
        self.extremaLines.clear()

        all_extrema = []

        if self.extremaEnabled.get():
            range_min = self.extremaRangeMin.get() * 1e9
            range_max = self.extremaRangeMax.get() * 1e9
            
            for ax, prm in zip(axes, sel):
                for v, p, d in self.fls:
                    if not v.get(): continue
                    ext = Path(p).suffix.lower()
                    if d.get('is_average', False) and d.get('custom_name'):
                        lbl = d['custom_name']
                    else:
                        lbl = Path(p).stem
                    try:
                        if d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']:
                            ntw_full = d.get('ntwk_full')
                            if ntw_full is None:
                                ntw_full = loadFile(p)
                                d['ntwk_full'] = ntw_full
                            
                            ntw = ntw_full[sstr]
                            arr = getattr(ntw, prm).s_db.flatten()
                            fr = ntw.f
                            
                            mask = (fr >= range_min) & (fr <= range_max)
                            if np.any(mask):
                                fr_masked = fr[mask]
                                arr_masked = arr[mask]
                                
                                if self.extremaMaxima.get():
                                    max_idx = np.argmax(arr_masked)
                                    max_freq = fr_masked[max_idx]
                                    max_val = arr_masked[max_idx]
                                    line = ax.axvline(x=max_freq, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
                                    self.extremaLines.append(line)
                                    all_extrema.append({
                                        'type': 'max',
                                        'freq': max_freq,
                                        'value': max_val,
                                        'param': prm,
                                        'file': lbl
                                    })
                                
                                if self.extremaMinima.get():
                                    min_idx = np.argmin(arr_masked)
                                    min_freq = fr_masked[min_idx]
                                    min_val = arr_masked[min_idx]
                                    line = ax.axvline(x=min_freq, color='blue', linestyle='--', alpha=0.3, linewidth=0.8)
                                    self.extremaLines.append(line)
                                    all_extrema.append({
                                        'type': 'min',
                                        'freq': min_freq,
                                        'value': min_val,
                                        'param': prm,
                                        'file': lbl
                                    })
                        
                        elif ext == '.csv' and prm == 's11':
                            if 'csv_data' not in d:
                                df = pd.read_csv(p)
                                d['csv_data'] = (df.iloc[:,0].values, df.iloc[:,1].values)
                            fr, arr = d['csv_data']
                            
                            mask = (fr >= range_min) & (fr <= range_max)
                            if np.any(mask):
                                fr_masked = fr[mask]
                                arr_masked = arr[mask]
                                
                                if self.extremaMaxima.get():
                                    max_idx = np.argmax(arr_masked)
                                    max_freq = fr_masked[max_idx]
                                    max_val = arr_masked[max_idx]
                                    line = ax.axvline(x=max_freq, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
                                    self.extremaLines.append(line)
                                    all_extrema.append({
                                        'type': 'max',
                                        'freq': max_freq,
                                        'value': max_val,
                                        'param': prm,
                                        'file': lbl
                                    })
                                
                                if self.extremaMinima.get():
                                    min_idx = np.argmin(arr_masked)
                                    min_freq = fr_masked[min_idx]
                                    min_val = arr_masked[min_idx]
                                    line = ax.axvline(x=min_freq, color='blue', linestyle='--', alpha=0.3, linewidth=0.8)
                                    self.extremaLines.append(line)
                                    all_extrema.append({
                                        'type': 'min',
                                        'freq': min_freq,
                                        'value': min_val,
                                        'param': prm,
                                        'file': lbl
                                    })
                    except Exception:
                        pass
            
            if all_extrema:
                self.txt.config(state=tk.NORMAL)
                self.txt.insert(tk.END, "\n" + "="*40 + "\n")
                self.txt.insert(tk.END, f"EXTREMA (Range: {self.extremaRangeMin.get():.2f}-{self.extremaRangeMax.get():.2f} GHz)\n")
                self.txt.insert(tk.END, "-"*40 + "\n")
                
                all_extrema.sort(key=lambda x: (x['param'], x['type'], x['freq']))
                
                for ext in all_extrema:
                    type_str = "MAX" if ext['type'] == 'max' else "MIN"
                    freq_ghz = ext['freq'] / 1e9
                    self.txt.insert(tk.END, f"{type_str} | {ext['file']} | {ext['param'].upper()} | {freq_ghz:.4f} GHz | {ext['value']:.2f} dB\n")
                
                self.txt.config(state=tk.DISABLED)

        axes[-1].set_xlabel("Frequency [Hz]")
        self.fig.tight_layout(rect=[0, 0, 1, 1])
        self.cv.draw()
        
        self._updateLegendPanel(all_legend_items)

    def _updTPlot(self):
        self.figT.clf()
        prm = self.td.get()
        center = self.gateCenter.get()
        span = self.gateSpan.get()
        doGate = self.gateChk.get()
        fmin = self.fmin.get()
        fmax = self.fmax.get()
        sstr = f"{fmin}-{fmax}ghz"
        
        axes = self.figT.subplots(3, 1, sharex=False)
        self.tax = list(axes)
        
        for ax in axes:
            ax.set_navigate(True)
        
        axF = axes[0]
        axTR = axes[1]
        axTG = axes[2]
        legF = []
        legTR = []
        legTG = []
        legend_items = []  
        legend_colors = {}  
        
        for v, p, d in self.fls:
            if not v.get(): continue
            ext = Path(p).suffix.lower()
            if d.get('is_average', False) and d.get('custom_name'):
                lbl = d['custom_name']
            else:
                lbl = Path(p).stem
            try:
                if d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']:
                    ntw_full = d.get('ntwk_full')
                    cached_range = d.get('cached_range')
                    
                    if ntw_full is None:
                        ntw_full = loadFile(p)
                        d['ntwk_full'] = ntw_full
                    
                    if cached_range != sstr:
                        ntw = ntw_full[sstr]
                        d['ntwk'] = ntw
                        d['cached_range'] = sstr
                    else:
                        ntw = d['ntwk']
                    
                    arr = getattr(ntw, prm).s_db.flatten()
                    fr = ntw.f
                    
                    line_kwargs = {'label': lbl}
                    if d.get('line_color'):
                        line_kwargs['color'] = d['line_color']
                    if d.get('line_width', 1.0) != 1.0:
                        line_kwargs['linewidth'] = d['line_width']
                    
                    line = axF.plot(fr, arr, **line_kwargs)[0]
                    legF.append(lbl)
                    legend_colors[lbl] = line.get_color()
                    
                elif ext == '.csv' and prm == 's11':
                    if 'csv_data' not in d:
                        df = pd.read_csv(p)
                        d['csv_data'] = (df.iloc[:,0].values, df.iloc[:,1].values)
                    fr, arr = d['csv_data']
                    
                    mask = (fr >= fmin * 1e9) & (fr <= fmax * 1e9)
                    fr = fr[mask]
                    arr = arr[mask]
                    
                    line_kwargs = {'label': lbl}
                    if d.get('line_color'):
                        line_kwargs['color'] = d['line_color']
                    if d.get('line_width', 1.0) != 1.0:
                        line_kwargs['linewidth'] = d['line_width']
                    
                    line = axF.plot(fr, arr, **line_kwargs)[0]
                    legF.append(lbl)
                    legend_colors[lbl] = line.get_color()
            except Exception:
                pass
        
        axF.set_ylabel(f"|{prm.upper()}| [dB]")
        axF.set_title(f"{prm.upper()} (frequency domain - raw)")
        axF.grid(True)
        if axF.get_legend():
            axF.get_legend().remove()

        if self.legendOnPlot.get() and legF:  

            ncol = min(5, (len(legF) - 1) // 10 + 1)
            axF.legend(legF, loc='best', ncol=ncol, fontsize=8, framealpha=0.9)

        if not legF:
            axF.text(0.5, 0.5, "Brak danych", ha="center", va="center", color="gray")
            axF.set_xticks([]); axF.set_yticks([])
        
        for v, p, d in self.fls:
            if not v.get(): continue
            ext = Path(p).suffix.lower()
            if d.get('is_average', False) and d.get('custom_name'):
                lbl = d['custom_name']
            else:
                lbl = Path(p).stem
            try:
                if ext in ['.s1p', '.s2p', '.s3p'] or d.get('is_average', True):
                    ntw_full = d.get('ntwk_full')
                    if ntw_full is None:
                        ntw_full = loadFile(p)
                        d['ntwk_full'] = ntw_full
                    
                    ntw = ntw_full[sstr]
                    
                    s_param = getattr(ntw, prm)
                    t_ns = s_param.frequency.t_ns
                    s_time_db = s_param.s_time_db.flatten()
                    
                    line_kwargs = {'label': lbl}
                    if d.get('line_color'):
                        line_kwargs['color'] = d['line_color']
                    elif lbl in legend_colors:
                        line_kwargs['color'] = legend_colors[lbl]
                    if d.get('line_width', 1.0) != 1.0:
                        line_kwargs['linewidth'] = d['line_width']
                    
                    line = axTR.plot(t_ns, s_time_db, **line_kwargs)[0]
                    legTR.append(lbl)
                    
                    if lbl not in legend_colors:
                        legend_colors[lbl] = line.get_color()
            except Exception:
                pass
        
        axTR.set_ylabel(f"{prm.upper()} TD [dB]")
        axTR.set_xlabel("Czas [ns]")
        axTR.set_title(f"{prm.upper()} (time domain - raw)")
        axTR.grid(True)
        axTR.set_xlim(0, 50)
        if axTR.get_legend():
            axTR.get_legend().remove()
        if self.legendOnPlot.get() and legTR:  
            ncol = min(5, (len(legTR) - 1) // 10 + 1)
            axTR.legend(legTR, loc='best', ncol=ncol, fontsize=8, framealpha=0.9)

        if not legTR:
            axTR.plot([0, 30], [0, 0], alpha=0)
            axTR.text(0.5, 0.5, "Brak danych / Placeholder", ha="center", va="center", color="gray", transform=axTR.transAxes)
            axTR.set_xticks(np.linspace(0, 30, 7))
            axTR.set_yticks([])
        
        if doGate:
            for v, p, d in self.fls:
                if not v.get(): continue
                ext = Path(p).suffix.lower()
                if d.get('is_average', False) and d.get('custom_name'):
                    lbl = d['custom_name']
                else:
                    lbl = Path(p).stem if not p.startswith("<") else p[1:-1]
                try:
                    if ext in ['.s1p', '.s2p', '.s3p'] or d.get('is_average', True):
                        ntw_full = d.get('ntwk_full')
                        if ntw_full is None:
                            ntw_full = loadFile(p)
                            d['ntwk_full'] = ntw_full
                        
                        ntw = ntw_full[sstr]
                        
                        sRaw = getattr(ntw, prm)
                        sGate = sRaw.time_gate(center=center, span=span)
                        freq = ntw.f
                        arr = sGate.s_db.flatten()
                        
                        line_kwargs = {'label': lbl}
                        if d.get('line_color'):
                            line_kwargs['color'] = d['line_color']
                        elif lbl in legend_colors:
                            line_kwargs['color'] = legend_colors[lbl]
                        if d.get('line_width', 1.0) != 1.0:
                            line_kwargs['linewidth'] = d['line_width']
                        
                        line = axTG.plot(freq, arr, **line_kwargs)[0]
                        legTG.append(lbl)
                        
                        if lbl not in legend_colors:
                            legend_colors[lbl] = line.get_color()
                except Exception:
                    pass
        
        axTG.set_ylabel(f"{prm.upper()} [dB]")
        axTG.set_xlabel("Frequency [Hz]")
        axTG.set_title(f"{prm.upper()} (frequency domain - gated)")
        axTG.grid(True)
        if axTG.get_legend():
            axTG.get_legend().remove()
        if self.legendOnPlot.get() and legTG: 
            ncol = min(5, (len(legTG) - 1) // 10 + 1)
            axTG.legend(legTG, loc='best', ncol=ncol, fontsize=8, framealpha=0.9)
        if not legTG:
            axTG.text(0.5, 0.5, "Brak danych / Placeholder", ha="center", va="center", color="gray", transform=axTG.transAxes)
            axTG.set_xticks([]); axTG.set_yticks([])
        

        for line in self.extremaLinesT:
            try:
                line.remove()
            except:
                pass
        self.extremaLinesT.clear()

        all_extrema = []

        if self.extremaEnabled.get():
            range_min = self.extremaRangeMin.get() * 1e9
            range_max = self.extremaRangeMax.get() * 1e9
            
            for ax, ax_label in [(axF, 'raw'), (axTG, 'gated')]:
                if ax_label == 'gated' and not doGate:
                    continue
                    
                for v, p, d in self.fls:
                    if not v.get(): continue
                    ext = Path(p).suffix.lower()
                    if d.get('is_average', False) and d.get('custom_name'):
                        lbl = d['custom_name']
                    else:
                        lbl = Path(p).stem
                    try:
                        if d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']:
                            ntw_full = d.get('ntwk_full')
                            if ntw_full is None:
                                ntw_full = loadFile(p)
                                d['ntwk_full'] = ntw_full
                            
                            ntw = ntw_full[sstr]
                            
                            if ax_label == 'raw':
                                arr = getattr(ntw, prm).s_db.flatten()
                                fr = ntw.f
                            else:
                                sRaw = getattr(ntw, prm)
                                sGate = sRaw.time_gate(center=center, span=span)
                                arr = sGate.s_db.flatten()
                                fr = ntw.f
                            
                            mask = (fr >= range_min) & (fr <= range_max)
                            if np.any(mask):
                                fr_masked = fr[mask]
                                arr_masked = arr[mask]
                                
                                if self.extremaMaxima.get():
                                    max_idx = np.argmax(arr_masked)
                                    max_freq = fr_masked[max_idx]
                                    max_val = arr_masked[max_idx]
                                    line = ax.axvline(x=max_freq, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
                                    self.extremaLinesT.append(line)
                                    all_extrema.append({
                                        'type': 'max',
                                        'freq': max_freq,
                                        'value': max_val,
                                        'param': f'{prm}_{ax_label}',
                                        'file': lbl
                                    })
                                
                                if self.extremaMinima.get():
                                    min_idx = np.argmin(arr_masked)
                                    min_freq = fr_masked[min_idx]
                                    min_val = arr_masked[min_idx]
                                    line = ax.axvline(x=min_freq, color='blue', linestyle='--', alpha=0.3, linewidth=0.8)
                                    self.extremaLinesT.append(line)
                                    all_extrema.append({
                                        'type': 'min',
                                        'freq': min_freq,
                                        'value': min_val,
                                        'param': f'{prm}_{ax_label}',
                                        'file': lbl
                                    })
                        
                        elif ext == '.csv' and prm == 's11' and ax_label == 'raw':
                            if 'csv_data' not in d:
                                df = pd.read_csv(p)
                                d['csv_data'] = (df.iloc[:,0].values, df.iloc[:,1].values)
                            fr, arr = d['csv_data']
                            
                            mask = (fr >= range_min) & (fr <= range_max)
                            if np.any(mask):
                                fr_masked = fr[mask]
                                arr_masked = arr[mask]
                                
                                if self.extremaMaxima.get():
                                    max_idx = np.argmax(arr_masked)
                                    max_freq = fr_masked[max_idx]
                                    max_val = arr_masked[max_idx]
                                    line = ax.axvline(x=max_freq, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
                                    self.extremaLinesT.append(line)
                                    all_extrema.append({
                                        'type': 'max',
                                        'freq': max_freq,
                                        'value': max_val,
                                        'param': f'{prm}_{ax_label}',
                                        'file': lbl
                                    })
                                
                                if self.extremaMinima.get():
                                    min_idx = np.argmin(arr_masked)
                                    min_freq = fr_masked[min_idx]
                                    min_val = arr_masked[min_idx]
                                    line = ax.axvline(x=min_freq, color='blue', linestyle='--', alpha=0.3, linewidth=0.8)
                                    self.extremaLinesT.append(line)
                                    all_extrema.append({
                                        'type': 'min',
                                        'freq': min_freq,
                                        'value': min_val,
                                        'param': f'{prm}_{ax_label}',
                                        'file': lbl
                                    })
                    except Exception:
                        pass
            
            if all_extrema:
                self.txt.config(state=tk.NORMAL)
                self.txt.insert(tk.END, "\n" + "="*40 + "\n")
                self.txt.insert(tk.END, f"EXTREMA (Range: {self.extremaRangeMin.get():.2f}-{self.extremaRangeMax.get():.2f} GHz)\n")
                self.txt.insert(tk.END, "-"*40 + "\n")
                
                all_extrema.sort(key=lambda x: (x['param'], x['type'], x['freq']))
                
                for ext in all_extrema:
                    type_str = "MAX" if ext['type'] == 'max' else "MIN"
                    freq_ghz = ext['freq'] / 1e9
                    self.txt.insert(tk.END, f"{type_str} | {ext['file']} | {ext['param'].upper()} | {freq_ghz:.4f} GHz | {ext['value']:.2f} dB\n")
                
                self.txt.config(state=tk.DISABLED)
        self.figT.tight_layout()
        self.cvT.draw()
        
        if os.name == 'nt' and hasattr(self, 'tbT'):
            self.tbT.update()
            if not self.tbT.winfo_ismapped():
                self.tbT.pack(side=tk.BOTTOM, fill=tk.X)
        
        all_files = set(legF + legTR + legTG)
        for lbl in sorted(all_files):
            if lbl in legend_colors:
                legend_items.append((lbl, legend_colors[lbl]))
        
        self._updateLegendPanel(legend_items, 'time')

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
        if self.tab == "freq":
            self.cv.draw()
        elif self.tab == "time":
            self.cvT.draw()

    def _updAll(self):
        if self.tab == "freq":
            self._updPlot()
        elif self.tab == "time":
            self._updTPlot()
        elif self.tab == "regex":
            self._updRegexPlot()
        elif self.tab == "overlap":
            self._updOverlapPlot()
        elif self.tab == "variance":
            self._updVariancePlot()

    def _onClick(self, ev):
        if self.tab == "freq" and not self.markersEnabled.get():
            return
        elif self.tab == "time" and not self.markersEnabledTime.get():
            return
            
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
            
        avg_count = sum(1 for i in toDelete if self.fls[i][2].get('is_average', False))
        file_count = len(toDelete) - avg_count
        
        msg = f"Remove {len(toDelete)} items from the list?"
        if avg_count > 0:
            msg = f"Remove {file_count} files and {avg_count} averages from the list?"
            
        if messagebox.askyesno("Confirm deletion", msg):
            remaining_files = []
            for i, file_data in enumerate(self.fls):
                if i not in toDelete:
                    remaining_files.append(file_data)
            
            self.fls = remaining_files
            
            for widget in self.fbox.winfo_children():
                widget.destroy()
            
            for v, p, d in self.fls:
                v.set(True)
                if d.get('is_average', False):
                    name = Path(p).name if not p.startswith("<") else f"[AVG] {p[1:-1].replace('average_', 'average_')}"
                    chk = tk.Checkbutton(self.fbox, text=name, variable=v, command=self._updAll, fg="blue", activeforeground="blue")
                    chk.pack(anchor="w")
                else:
                    chk = ttk.Checkbutton(self.fbox, text=Path(p).name, variable=v, command=self._updAll)
                    chk.pack(anchor="w")
                chk.bind("<Button-3>", lambda e, path=p: self._showStyleMenu(e, path))
            
            self.fileListCanvas.configure(scrollregion=self.fileListCanvas.bbox("all"))
            self._updAll()
            messagebox.showinfo("Files removed", f"Removed {len(toDelete)} items from the list")

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
        
        dialog = tk.Toplevel(self)
        dialog.title("Name this import")
        dialog.geometry("300x100")
        dialog.transient(self)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Import name:").pack(pady=10)
        name_var = tk.StringVar(value=f"regex_{self.regexParam.get()}_{len(self.overlapData)+1}")
        entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        entry.pack(pady=5)
        entry.select_range(0, tk.END)
        entry.focus()
        
        def ok():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("Invalid name", "Please enter a name")
                return
            if name in self.overlapData:
                if not messagebox.askyesno("Overwrite?", f"'{name}' already exists. Overwrite?"):
                    return
            self.overlapData[name] = [(s/1e9, e/1e9) for s, e in self.regexRanges]
            dialog.destroy()
            self._updOverlapPlot()
        
        def cancel():
            dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
        
        entry.bind("<Return>", lambda e: ok())
        entry.bind("<Escape>", lambda e: cancel())
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

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
        
        self.txt.insert(tk.END, "\nOverlapping ranges:\n")
        self.txt.insert(tk.END, "-" * 40 + "\n")
        for (n1, r1), (n2, r2) in combinations(self.overlapData.items(), 2):
            overlaps = self._findOverlaps(r1, r2)
            if overlaps:
                formatted = [(round(s, 3), round(e, 3)) for s, e in overlaps]
                self.txt.insert(tk.END, f"{n1} <-> {n2}: {formatted}\n")
            else:
                self.txt.insert(tk.END, f"{n1} <-> {n2}: no overlaps\n")
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

    def _updTDAnalysisPlot(self):
        self.axTDA1.clear()
        self.axTDA2.clear()
        
        fmin = self.fmin.get()
        fmax = self.fmax.get()
        sstr = f"{fmin}-{fmax}ghz"
        time_limit = self.tdAnalysisTimeLimit.get()
        
        analysis_results = []
        
        for v, p, d in self.fls:
            if not v.get():
                continue
                
            ext = Path(p).suffix.lower()
            if d.get('is_average', False) and d.get('custom_name'):
                lbl = d['custom_name']
            else:
                lbl = Path(p).stem if not p.startswith("<") else p[1:-1]
                
            if d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']:
                try:
                    ntw_full = d.get('ntwk_full')
                    cached_range = d.get('cached_range')
                    
                    if ntw_full is None:
                        ntw_full = loadFile(p)
                        d['ntwk_full'] = ntw_full
                    
                    if cached_range != sstr:
                        ntw = ntw_full[sstr]
                        d['ntwk'] = ntw
                        d['cached_range'] = sstr
                    else:
                        ntw = d['ntwk']
                    
                    s11_td = ntw.s11
                    t_ns = s11_td.frequency.t_ns
                    s11_time_db = s11_td.s_time_db.flatten()
                    
                    if ntw.nports >= 2:
                        s21_td = ntw.s21
                        s21_time_db = s21_td.s_time_db.flatten()
                    else:
                        s21_time_db = None
                    
                    time_mask = (t_ns >= 0) & (t_ns <= time_limit)
                    t_ns_limited = t_ns[time_mask]
                    s11_limited = s11_time_db[time_mask]
                    
                    line_kwargs = {'label': lbl}
                    if d.get('line_color'):
                        line_kwargs['color'] = d['line_color']
                    if d.get('line_width', 1.0) != 1.0:
                        line_kwargs['linewidth'] = d['line_width']
                    
                    self.axTDA1.plot(t_ns_limited, s11_limited, **line_kwargs)
                    
                    s11_max_idx = np.argmax(s11_limited)
                    s11_max_time = t_ns_limited[s11_max_idx]
                    s11_max_val = s11_limited[s11_max_idx]
                    
                    if s21_time_db is not None:
                        s21_limited = s21_time_db[time_mask]
                        self.axTDA2.plot(t_ns_limited, s21_limited, **line_kwargs)
                        
                        s21_max_idx = np.argmax(s21_limited)
                        s21_max_time = t_ns_limited[s21_max_idx]
                        s21_max_val = s21_limited[s21_max_idx]
                        
                        condition_met = abs(s11_max_time - 2*s21_max_time) <= 0.5
                        
                        analysis_results.append({
                            'file': lbl,
                            's11_max_time': s11_max_time,
                            's11_max_val': s11_max_val,
                            's21_max_time': s21_max_time,
                            's21_max_val': s21_max_val,
                            'condition_met': condition_met
                        })
                        
                        if self.tdAnalysisMarkPeaks.get():
                            color = 'green' if condition_met else 'red'
                            self.axTDA1.plot(s11_max_time, s11_max_val, 'o', color=color, 
                                           markersize=8, markeredgecolor='black', markeredgewidth=1)
                            self.axTDA2.plot(s21_max_time, s21_max_val, 'o', color=color,
                                           markersize=8, markeredgecolor='black', markeredgewidth=1)
                    else:
                        analysis_results.append({
                            'file': lbl,
                            's11_max_time': s11_max_time,
                            's11_max_val': s11_max_val,
                            's21_max_time': None,
                            's21_max_val': None,
                            'condition_met': None
                        })
                        
                        if self.tdAnalysisMarkPeaks.get():
                            self.axTDA1.plot(s11_max_time, s11_max_val, 'o', color='gray',
                                           markersize=8, markeredgecolor='black', markeredgewidth=1)
                    
                except Exception as e:
                    pass
        
        self.tdAnalysisData = analysis_results
        
        self.axTDA1.set_ylabel('S11 [dB]')
        self.axTDA1.set_title('S11 Time Domain')
        self.axTDA1.grid(True, alpha=0.3)
        self.axTDA1.set_xlim(0, time_limit)
        
        self.axTDA2.set_ylabel('S21 [dB]')
        self.axTDA2.set_xlabel('Time [ns]')
        self.axTDA2.set_title('S21 Time Domain')
        self.axTDA2.grid(True, alpha=0.3)
        self.axTDA2.set_xlim(0, time_limit)
        
        if self.legendOnPlot.get():
            handles1, labels1 = self.axTDA1.get_legend_handles_labels()
            if labels1:
                ncol = min(5, (len(labels1) - 1) // 10 + 1)
                self.axTDA1.legend(loc='best', ncol=ncol, fontsize=8, framealpha=0.9)
            
            handles2, labels2 = self.axTDA2.get_legend_handles_labels()
            if labels2:
                ncol = min(5, (len(labels2) - 1) // 10 + 1)
                self.axTDA2.legend(loc='best', ncol=ncol, fontsize=8, framealpha=0.9)
        
        self.figTDA.tight_layout()
        self.cvTDA.draw()
        
        self.txt.config(state=tk.NORMAL)
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, "Time Domain Analysis\n")
        self.txt.insert(tk.END, "="*40 + "\n")
        self.txt.insert(tk.END, f"Condition: t_max(S11) == 2 * t_max(S21) (+/- 0.5ns)\n")
        self.txt.insert(tk.END, f"Time window: 0 - {time_limit} ns\n")
        self.txt.insert(tk.END, "-"*40 + "\n\n")
        
        for res in analysis_results:
            self.txt.insert(tk.END, f"{res['file']}:\n")
            self.txt.insert(tk.END, f"  S11 max: {res['s11_max_time']:.3f} ns @ {res['s11_max_val']:.2f} dB\n")
            
            if res['s21_max_time'] is not None:
                self.txt.insert(tk.END, f"  S21 max: {res['s21_max_time']:.3f} ns @ {res['s21_max_val']:.2f} dB\n")
                self.txt.insert(tk.END, f"  2 * t_max(S11) = {2*res['s11_max_time']:.3f} ns\n")
                
                if res['condition_met']:
                    self.txt.insert(tk.END, f"  Condition: PASS (GREEN)\n")
                else:
                    self.txt.insert(tk.END, f"  Condition: FAIL (RED)\n")
            else:
                self.txt.insert(tk.END, f"  S21: Not available (1-port device)\n")
            
            self.txt.insert(tk.END, "\n")
        
        for txt in self.tdAnalysisMtxt:
            self.txt.insert(tk.END, txt)
        
        self.txt.config(state=tk.DISABLED)


    def _onClickTDAnalysis(self, event):
        if getattr(self, "_blkNextTDA", False):
            self._blkNextTDA = False
            return
        
        if event.inaxes not in [self.axTDA1, self.axTDA2] or self.tdAnalysisData is None:
            return
        
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return
        
        # Find closest point
        ax = event.inaxes
        mind = float("inf")
        best = None
        
        for ln in ax.get_lines():
            if ln.get_label().startswith('_'):
                continue
            xd = ln.get_xdata()
            yd = ln.get_ydata()
            dists = np.sqrt((xd - x)**2 + (yd - y)**2)
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
            m, = ax.plot([xf], [yf], 'o', color=col, markersize=10, picker=7)
            
            param = "S11" if ax == self.axTDA1 else "S21"
            t = ax.annotate(
                f"{xf:.3f} ns\n{yf:.2f} dB",
                xy=(xf, yf), xytext=(10, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.5),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
                fontsize=9
            )
            t.set_picker(True)
            
            self.tdAnalysisMrk.append({'marker': m, 'text': t, 'ax': ax})
            
            txt = f"{lbl} {param}: {xf:.3f} ns @ {yf:.2f} dB\n"
            self.tdAnalysisMtxt.append(txt)
            self.txt.config(state=tk.NORMAL)
            self.txt.insert(tk.END, txt)
            self.txt.config(state=tk.DISABLED)
            
            self.cvTDA.draw()



    def _clearTDAnalysisMarkers(self):
        for m in self.tdAnalysisMrk:
            try:
                m['marker'].remove()
                m['text'].remove()
            except:
                pass
        self.tdAnalysisMrk.clear()
        self.tdAnalysisMtxt.clear()
        
        # Refresh text
        if self.tdAnalysisData:
            self._updTDAnalysisPlot()
        else:
            self.txt.config(state=tk.NORMAL)
            self.txt.delete(1.0, tk.END)
            self.txt.config(state=tk.DISABLED)
        
        self.cvTDA.draw()


    def _exportTDAnalysis(self):
        if not self.tdAnalysisData:
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
                for res in self.tdAnalysisData:
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
                f.write("Time Domain Analysis Results\n")
                f.write("="*60 + "\n")
                f.write(f"Condition: t_max(S21) > 2 * t_max(S11)\n")
                f.write(f"Time window: 0 - {self.tdAnalysisTimeLimit.get()} ns\n")
                f.write(f"Frequency range: {self.fmin.get()}-{self.fmax.get()} GHz\n")
                f.write("-"*60 + "\n\n")
                
                for res in self.tdAnalysisData:
                    f.write(f"File: {res['file']}\n")
                    f.write(f"  S11 max: {res['s11_max_time']:.3f} ns @ {res['s11_max_val']:.2f} dB\n")
                    
                    if res['s21_max_time'] is not None:
                        f.write(f"  S21 max: {res['s21_max_time']:.3f} ns @ {res['s21_max_val']:.2f} dB\n")
                        f.write(f"  2 * t_max(S11) = {2*res['s11_max_time']:.3f} ns\n")
                        f.write(f"  Condition: {'PASS' if res['condition_met'] else 'FAIL'}\n")
                    else:
                        f.write(f"  S21: Not available\n")
                    f.write("\n")
                
                # Add summary
                valid_results = [r for r in self.tdAnalysisData if r['condition_met'] is not None]
                if valid_results:
                    passed = sum(1 for r in valid_results if r['condition_met'])
                    total = len(valid_results)
                    f.write("-"*60 + "\n")
                    f.write(f"Summary: {passed}/{total} files pass condition\n")
                    f.write(f"Pass rate: {100*passed/total:.1f}%\n")
        
        messagebox.showinfo("Export complete", f"Analysis saved to {filename}")


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
            
            f.write("\nOverlapping ranges:\n\n")
            for (n1, r1), (n2, r2) in combinations(self.overlapData.items(), 2):
                overlaps = self._findOverlaps(r1, r2)
                if overlaps:
                    formatted = [(round(s, 3), round(e, 3)) for s, e in overlaps]
                    f.write(f"{n1} <-> {n2}: {formatted}\n")
                else:
                    f.write(f"{n1} <-> {n2}: no overlaps\n")
        
        messagebox.showinfo("Export complete", f"Overlap analysis saved to {filename}")

    def _clearOverlapPlot(self):
        self.overlapData = {}
        self._updOverlapPlot()

    def _normalizeForVariance(self, data, is_phase=False):
        if is_phase and self.varDetrend.get():
            n_samples, n_frequencies = data.shape
            detrended = np.zeros_like(data)
            
            for i in range(n_samples):
                x = np.arange(n_frequencies)
                coeffs = np.polyfit(x, data[i], 1)
                trend = np.polyval(coeffs, x)
                detrended[i] = data[i] - trend
            
            mean = np.mean(detrended)
            std = np.std(detrended) + 1e-10
            return (detrended - mean) / std
        else:
            mean = np.mean(data)
            std = np.std(data) + 1e-10
            return (data - mean) / std

    def _calculateVariance(self):
        fmin = self.fmin.get()
        fmax = self.fmax.get()
        sstr = f"{fmin}-{fmax}ghz"
        
        param_list = []
        component_types = []
        
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
        
        raw_data_by_param = {param: [] for param in param_list}
        file_count = 0
        freq_ref = None
        
        for v, p, d in self.fls:
            if not v.get():
                continue
            ext = Path(p).suffix.lower()
            if d.get('is_average', False) and d.get('custom_name'):
                lbl = d['custom_name']
            else:
                lbl = Path(p).stem if not p.startswith("<") else p[1:-1]
            if d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']:

                try:
                    ntw_full = d.get('ntwk_full')
                    cached_range = d.get('cached_range')
                    
                    if ntw_full is None:
                        ntw_full = loadFile(p)
                        d['ntwk_full'] = ntw_full
                    
                    if cached_range != sstr:
                        ntw = ntw_full[sstr]
                        d['ntwk'] = ntw
                        d['cached_range'] = sstr
                    else:
                        ntw = d['ntwk']
                    
                    if freq_ref is None:
                        freq_ref = ntw.f
                    elif len(ntw.f) != len(freq_ref) or not np.allclose(ntw.f, freq_ref, rtol=1e-6):
                        continue
                    
                    for sparam in ['s11', 's12', 's21', 's22']:
                        if any(sparam in p for p in param_list):
                            s_data = getattr(ntw, sparam)
                            
                            if f'{sparam}_mag' in param_list:
                                raw_data_by_param[f'{sparam}_mag'].append(s_data.s_db.flatten())
                            
                            if f'{sparam}_phase' in param_list:
                                phase_rad = np.angle(s_data.s.flatten())
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
        
        normalized_data = np.zeros((file_count, n_frequencies, n_params))
        
        for i, (param, comp_type) in enumerate(zip(param_list, component_types)):
            param_data = np.array(raw_data_by_param[param])
            
            normalized = self._normalizeForVariance(param_data, is_phase=(comp_type == 'phase'))
            normalized_data[:, :, i] = normalized
        
        variance_by_param = np.var(normalized_data, axis=0, ddof=1)
        total_variance = np.sum(variance_by_param, axis=1)
        
        variance_contribution = np.zeros_like(variance_by_param)
        for freq_idx in range(n_frequencies):
            if total_variance[freq_idx] > 1e-10:
                variance_contribution[freq_idx] = (variance_by_param[freq_idx] / 
                                                 total_variance[freq_idx]) * 100
        
        mean_by_param = np.mean(normalized_data, axis=0)
        std_by_param = np.std(normalized_data, axis=0, ddof=1)
        
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
            parts = param.split('_')
            s_param = parts[0].upper()
            comp_type = parts[1].capitalize()
            param_labels.append(f'{s_param} {comp_type}')
        
        y = np.zeros((n_params, len(freqs)))
        for i in range(n_params):
            y[i] = var_contrib[:, i]
        
        if n_params <= 8:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'][:n_params]
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_params))
        
        self.axV.stackplot(freqs, y, labels=param_labels[:n_params], 
                          colors=colors, alpha=0.8)
        
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
        self.axV.set_ylim(0, 105)
        
        if n_params <= 8:
            self.axV.legend(loc='upper right', fontsize=9)
        
        self.varFreqs = freqs
        self.varTotalVar = self.varData['total_variance']
        
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
        
        self.figV.tight_layout(rect=[0, 0, 1, 0.95])
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
        
        y_pos = 95
        
        m, = self.axV.plot([freq], [y_pos], 'ro', markersize=10, picker=7)
        
        t = self.axV.annotate(
            f"{freq/1e6:.3f} MHz\nVar: {var:.4f}",
            xy=(freq, y_pos), xytext=(10, -30),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
            fontsize=9
        )
        t.set_picker(True)
        
        self.varMrk.append({'marker': m, 'text': t})
        
        txt = f"Variance marker: {freq/1e6:.3f} MHz\n"
        txt += f"  Total variance: {var:.4f}\n"
        
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

    def _addTrainDir(self):
        if not self.modelTrainingAvailable:
            return
        
        directory = filedialog.askdirectory(title="Select directory containing .s2p files")
        if directory and directory not in self.trainDataDirs:
            self.trainDataDirs.append(directory)
            self.trainDirListbox.insert(tk.END, directory)
            
    def _removeTrainDir(self):
        if not self.modelTrainingAvailable:
            return
            
        selection = self.trainDirListbox.curselection()
        if selection:
            idx = selection[0]
            self.trainDirListbox.delete(idx)
            del self.trainDataDirs[idx]
            
    def _clearTrainDirs(self):
        if not self.modelTrainingAvailable:
            return
            
        self.trainDataDirs.clear()
        self.trainDirListbox.delete(0, tk.END)
        
    def _runCrossValidation(self):
        if not self.modelTrainingAvailable:
            return
            
        if not self.trainDataDirs:
            messagebox.showwarning("No data", "Please add training data directories first")
            return
            
        thread = threading.Thread(target=self._runCrossValidationThread)
        thread.daemon = True
        thread.start()
        
    def _runCrossValidationThread(self):
            self.after(0, self.trainProgress.start, 10)
            
            def update_text():
                self.txt.config(state=tk.NORMAL)
                self.txt.delete(1.0, tk.END)
                self.txt.insert(tk.END, "Running cross-validation...\n")
                self.txt.insert(tk.END, "This may take several minutes.\n\n")
                self.txt.config(state=tk.DISABLED)
            self.after(0, update_text)
            
            try:
                device = self.trainDevice.get()
                if device == "cuda" and self.torch and not self.torch.cuda.is_available():
                    device = "cpu"
                    def warn_cuda():
                        self.txt.config(state=tk.NORMAL)
                        self.txt.insert(tk.END, "WARNING: CUDA not available, using CPU instead.\n\n")
                        self.txt.config(state=tk.DISABLED)
                    self.after(0, warn_cuda)
                
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                fold_results, avg_mae, std_mae = self.water_pred.cross_validate_water_predictor(
                    data_dirs=self.trainDataDirs,
                    device=device,
                    hidden_dim=self.trainHiddenDim.get(),
                    latent_dim=self.trainLatentDim.get(),
                    dropout=self.trainDropout.get(),
                    lr=self.trainLR.get(),
                    weight_decay=self.trainWeightDecay.get(),
                    loss_fn=self.trainLossFn.get(),
                    augment=self.trainAugment.get(),
                    noise_std=self.trainNoise.get(),
                    use_batchnorm=self.trainBatchNorm.get(),
                    epochs=self.trainEpochs.get(),
                    patience=self.trainPatience.get()
                )
                
                sys.stdout = old_stdout
                
                def update_results():
                    self._plotCrossValidationResults(fold_results, avg_mae, std_mae)
                    
                    self.txt.config(state=tk.NORMAL)
                    self.txt.insert(tk.END, f"\nCross-validation complete!\n")
                    self.txt.insert(tk.END, f"Average MAE: {avg_mae:.2f} +/- {std_mae:.2f} ml\n")
                    self.txt.insert(tk.END, "\nFold results:\n")
                    for i, result in enumerate(fold_results):
                        self.txt.insert(tk.END, f"  Fold {i+1}: MAE = {result['val_mae']:.2f} ml\n")
                    
                    if os.path.exists('water_prediction_results.png'):
                        self.txt.insert(tk.END, "\nPlot saved to: water_prediction_results.png\n")
                    self.txt.config(state=tk.DISABLED)
                    
                self.after(0, update_results)
                
            except Exception as e:
                error_msg = str(e)
                if "expected more than 1 value per channel" in error_msg:
                    error_msg = "Not enough data for cross-validation with batch normalization.\nTry disabling batch norm or adding more training data."
                self.after(0, messagebox.showerror, "Error", f"Cross-validation failed:\n{error_msg}")
                
            finally:
                self.after(0, self.trainProgress.stop)
            
    def _trainFinalModel(self):
        if not self.modelTrainingAvailable:
            return
            
        if not self.trainDataDirs:
            messagebox.showwarning("No data", "Please add training data directories first")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save model as",
            defaultextension=".pth",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
            
        thread = threading.Thread(target=self._trainFinalModelThread, args=(save_path,))
        thread.daemon = True
        thread.start()
        
    def _trainFinalModelThread(self, save_path):
        self.after(0, self.trainProgress.start, 10)
        
        def update_text():
            self.txt.config(state=tk.NORMAL)
            self.txt.delete(1.0, tk.END)
            self.txt.insert(tk.END, "Training final model...\n")
            self.txt.insert(tk.END, "This may take several minutes.\n\n")
            self.txt.config(state=tk.DISABLED)
        self.after(0, update_text)
        
        try:
            device = self.trainDevice.get()
            if device == "cuda" and self.torch and not self.torch.cuda.is_available():
                device = "cpu"
                def warn_cuda():
                    self.txt.config(state=tk.NORMAL)
                    self.txt.insert(tk.END, "WARNING: CUDA not available, using CPU instead.\n\n")
                    self.txt.config(state=tk.DISABLED)
                self.after(0, warn_cuda)
            
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            model, dataset = self.water_pred.train_final_model(
                data_dirs=self.trainDataDirs,
                model_save_path=save_path,
                device=device,
                hidden_dim=self.trainHiddenDim.get(),
                latent_dim=self.trainLatentDim.get(),
                dropout=self.trainDropout.get(),
                lr=self.trainLR.get(),
                weight_decay=self.trainWeightDecay.get(),
                loss_fn=self.trainLossFn.get(),
                augment=self.trainAugment.get(),
                noise_std=self.trainNoise.get(),
                use_batchnorm=self.trainBatchNorm.get(),
                epochs=self.trainEpochs.get(),
                patience=self.trainPatience.get()
            )
            
            sys.stdout = old_stdout
            
            def update_results():
                self.txt.config(state=tk.NORMAL)
                self.txt.insert(tk.END, f"\nModel training complete!\n")
                self.txt.insert(tk.END, f"Model saved to: {save_path}\n")
                
                if os.path.exists('final_model_training.png'):
                    self.txt.insert(tk.END, "Plot saved to: final_model_training.png\n")
                    
                    try:
                        from PIL import Image
                        img = Image.open('final_model_training.png')
                        
                        self.axM1.clear()
                        self.axM2.clear()
                        
                        self.axM1.imshow(img)
                        self.axM1.axis('off')
                        self.axM2.text(0.5, 0.5, "Training complete!\nSee final_model_training.png", 
                                      ha="center", va="center", fontsize=12)
                        self.axM2.set_xticks([])
                        self.axM2.set_yticks([])
                        
                        self.figM.tight_layout()
                        self.cvM.draw()
                    except:
                        pass
                
                self.txt.config(state=tk.DISABLED)
                
                messagebox.showinfo("Success", f"Model trained and saved to:\n{save_path}")
                
            self.after(0, update_results)
            
        except Exception as e:
            self.after(0, messagebox.showerror, "Error", f"Model training failed:\n{str(e)}")
            
        finally:
            self.after(0, self.trainProgress.stop)


    def _loadModel(self):
        if not self.modelTrainingAvailable:
            return
            
        filepath = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
            
        try:
            checkpoint = self.torch.load(filepath, map_location=self.trainDevice.get())
            
            self.txt.config(state=tk.NORMAL)
            self.txt.delete(1.0, tk.END)
            self.txt.insert(tk.END, f"Model loaded: {os.path.basename(filepath)}\n")
            self.txt.insert(tk.END, "=" * 40 + "\n")
            
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                self.txt.insert(tk.END, "Model configuration:\n")
                for key, value in config.items():
                    self.txt.insert(tk.END, f"  {key}: {value}\n")
                    
            self.txt.insert(tk.END, "\nModel ready for inference.\n")
            self.txt.config(state=tk.DISABLED)
            
            messagebox.showinfo("Success", "Model loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            
    def _plotCrossValidationResults(self, fold_results, avg_mae, std_mae):
        if not self.modelTrainingAvailable:
            return
            
        self.axM1.clear()
        self.axM2.clear()
        
        all_predictions = np.concatenate([r['predictions'] for r in fold_results])
        all_targets = np.concatenate([r['targets'] for r in fold_results])
        
        self.axM1.scatter(all_targets, all_predictions, alpha=0.6)
        self.axM1.plot([min(all_targets), max(all_targets)], 
                       [min(all_targets), max(all_targets)], 'r--')
        self.axM1.set_xlabel('True Water Volume (ml)')
        self.axM1.set_ylabel('Predicted Water Volume (ml)')
        self.axM1.set_title(f'Water Volume Prediction\nMAE: {avg_mae:.1f}+/-{std_mae:.1f} ml')
        self.axM1.grid(True, alpha=0.3)
        
        mae_per_fold = [r['val_mae'] for r in fold_results]
        folds = list(range(1, len(mae_per_fold) + 1))
        self.axM2.bar(folds, mae_per_fold)
        self.axM2.set_xlabel('Fold')
        self.axM2.set_ylabel('MAE (ml)')
        self.axM2.set_title('MAE by Fold')
        self.axM2.grid(True, alpha=0.3)
        
        self.axM2.axhline(y=avg_mae, color='r', linestyle='--', 
                          label=f'Average: {avg_mae:.1f} ml')
        self.axM2.legend()
        
        self.figM.tight_layout()
        self.cvM.draw()
        
    def _scanTrainingFiles(self):
        if not self.modelTrainingAvailable:
            return
            
        if not self.trainDataDirs:
            messagebox.showwarning("No data", "Please add training data directories first")
            return
            
        self.txt.config(state=tk.NORMAL)
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, "Scanning training files...\n")
        self.txt.insert(tk.END, "=" * 40 + "\n\n")
        
        total_files = 0
        valid_files = 0
        water_volumes = set()
        distances = set()
        combinations_count = {}
        
        for data_dir in self.trainDataDirs:
            self.txt.insert(tk.END, f"Directory: {data_dir}\n")
            
            s2p_files = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.lower().endswith('.s2p'):
                        s2p_files.append(os.path.join(root, file))
            
            self.txt.insert(tk.END, f"  Found {len(s2p_files)} .s2p files\n")
            total_files += len(s2p_files)
            
            dir_valid = 0
            for filepath in s2p_files:
                filename = os.path.basename(filepath)
                try:
                    water_match = re.search(r'(\d+)ml', filename.lower())
                    dist_mm_match = re.search(r'(\d+)mm', filename.lower())
                    dist_cm_match = re.search(r'(\d+)cm', filename.lower())
                    
                    if water_match and (dist_mm_match or dist_cm_match):
                        water_ml = float(water_match.group(1))
                        if dist_mm_match:
                            distance_mm = float(dist_mm_match.group(1))
                        else:
                            distance_cm = float(dist_cm_match.group(1))
                            distance_mm = distance_cm * 10.0
                        
                        water_volumes.add(water_ml)
                        distances.add(distance_mm)
                        dir_valid += 1
                        
                        combo_key = (water_ml, distance_mm)
                        combinations_count[combo_key] = combinations_count.get(combo_key, 0) + 1
                        
                except Exception:
                    pass
            
            self.txt.insert(tk.END, f"  Valid files: {dir_valid}\n\n")
            valid_files += dir_valid
        
        self.txt.insert(tk.END, "Summary:\n")
        self.txt.insert(tk.END, f"  Total .s2p files: {total_files}\n")
        self.txt.insert(tk.END, f"  Valid training files: {valid_files}\n")
        self.txt.insert(tk.END, f"  Unique water volumes: {len(water_volumes)}\n")
        self.txt.insert(tk.END, f"    Values (ml): {sorted(water_volumes)}\n")
        self.txt.insert(tk.END, f"  Unique distances: {len(distances)}\n")
        self.txt.insert(tk.END, f"    Values (mm): {sorted(distances)}\n")
        self.txt.insert(tk.END, f"    Values (cm): {[d/10 for d in sorted(distances)]}\n")
        self.txt.insert(tk.END, f"  Unique combinations: {len(combinations_count)}\n\n")
        
        if combinations_count:
            self.txt.insert(tk.END, "Data distribution:\n")
            for (water, dist), count in sorted(combinations_count.items()):
                self.txt.insert(tk.END, f"  {water}ml @ {dist}mm ({dist/10}cm): {count} files\n")
        
        if valid_files == 0:
            self.txt.insert(tk.END, "\nWARNING: No valid training files found!\n")
            self.txt.insert(tk.END, "Files must have format: ..._XXml_..._YYmm... or ..._XXml_..._YYcm...\n")
            self.txt.insert(tk.END, "where XX is water volume in ml and YY is distance in mm or cm\n")
        elif valid_files < 10:
            self.txt.insert(tk.END, f"\nWARNING: Only {valid_files} training files found.\n")
            self.txt.insert(tk.END, "Consider adding more data for better model performance.\n")
        
        self.txt.config(state=tk.DISABLED)

    def _cross_correlation(self, s1, s2, normalize=True, max_lag=None):
        if normalize:
            s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
            s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)
        
        n = len(s1) + len(s2) - 1
        S1 = np.fft.fft(s1, n)
        S2 = np.fft.fft(s2, n)
        corr = np.real(np.fft.ifft(S1 * np.conj(S2)))
        corr = np.concatenate([corr[-(len(s2)-1):], corr[:len(s1)]])
        
        lags = np.arange(-len(s2) + 1, len(s1))
        
        if max_lag is not None and max_lag > 0:
            valid_indices = np.abs(lags) <= max_lag
            corr = corr[valid_indices]
            lags = lags[valid_indices]
        
        best_idx = np.argmax(corr)
        
        if normalize:
            corr = corr / (np.sqrt(np.sum(s1**2) * np.sum(s2**2)) + 1e-10)
            
        return lags[best_idx], corr[best_idx]

    def _updCrossCorrPlot(self):
        self.figCC.clear()
        self.axCC = self.figCC.add_subplot(111)
        
        fmin = self.fmin.get()
        fmax = self.fmax.get()
        sstr = f"{fmin}-{fmax}ghz"
        param = self.crossCorrParam.get()
        normalize = self.crossCorrNormalize.get()
        
        files_data = []
        file_names = []
        
        for v, p, d in self.fls:
            if not v.get():
                continue
                
            ext = Path(p).suffix.lower()
            if ext not in ['.s1p', '.s2p', '.s3p']:
                continue
                
            try:
                ntw_full = d.get('ntwk_full')
                cached_range = d.get('cached_range')
                
                if ntw_full is None:
                    ntw_full = loadFile(p)
                    d['ntwk_full'] = ntw_full
                
                if cached_range != sstr:
                    ntw = ntw_full[sstr]
                    d['ntwk'] = ntw
                    d['cached_range'] = sstr
                else:
                    ntw = d['ntwk']
                
                s_data = getattr(ntw, param).s_db.flatten()
                freq = ntw.f
                
                files_data.append((s_data, freq))
                
                if d.get('is_average', False) and d.get('custom_name'):
                    filename_custom = d['custom_name']
                    file_names.append(filename_custom)
                else:
                    file_names.append(Path(p).stem)
                
            except Exception:
                pass
        
        n = len(files_data)
        
        if n < 2:
            self.axCC.text(0.5, 0.5, "Need at least 2 files selected", 
                          ha="center", va="center", fontsize=12, color="gray")
            self.axCC.set_xticks([])
            self.axCC.set_yticks([])
            self.cvCC.draw()
            return
        
        lag_matrix = np.zeros((n, n))
        corr_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    lag_matrix[i, j] = 0
                else:
                    lag, max_corr = self._cross_correlation(files_data[i][0], files_data[j][0], normalize)
                    corr_matrix[i, j] = corr_matrix[j, i] = max_corr
                    lag_matrix[i, j] = lag_matrix[j, i] = lag
        
        self.crossCorrData = {
            'corr_matrix': corr_matrix,
            'lag_matrix': lag_matrix,
            'file_names': file_names,
            'param': param,
            'normalized': normalize
        }
        
        im = self.axCC.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
        
        self.axCC.set_xticks(range(n))
        self.axCC.set_yticks(range(n))
        self.axCC.set_xticklabels(file_names, rotation=45, ha='right')
        self.axCC.set_yticklabels(file_names)
        
        for i in range(n):
            for j in range(n):
                text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
                text = self.axCC.text(j, i, f'{corr_matrix[i, j]:.2f}\n({int(lag_matrix[i, j])})',
                                     ha="center", va="center", color=text_color, fontsize=7)
        
        self.figCC.colorbar(im, ax=self.axCC, label='Cross-Correlation')
        
        title = f"Cross-Correlation Matrix - {param.upper()}"
        if normalize:
            title += " (Normalized)"
        self.axCC.set_title(title)
        
        self.figCC.tight_layout()
        self.cvCC.draw()
        
        self.txt.config(state=tk.NORMAL)
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, f"Cross-Correlation Analysis\n")
        self.txt.insert(tk.END, f"Parameter: {param.upper()}\n")
        self.txt.insert(tk.END, f"Mode: {'Normalized' if normalize else 'Raw'}\n")
        self.txt.insert(tk.END, f"Files analyzed: {n}\n")
        self.txt.insert(tk.END, "-" * 40 + "\n")
        
        corr_values = corr_matrix[np.triu_indices(n, k=1)]
        lag_values = lag_matrix[np.triu_indices(n, k=1)]
        
        self.txt.insert(tk.END, f"Max correlation: {np.max(corr_values):.3f}\n")
        self.txt.insert(tk.END, f"Min correlation: {np.min(corr_values):.3f}\n")
        self.txt.insert(tk.END, f"Mean correlation: {np.mean(corr_values):.3f}\n")
        self.txt.insert(tk.END, f"Mean lag: {np.mean(np.abs(lag_values)):.1f} bins\n")
        self.txt.insert(tk.END, "\nHighest correlations:\n")
        
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((corr_matrix[i, j], lag_matrix[i, j], file_names[i], file_names[j]))
        
        pairs.sort(reverse=True)
        for corr, lag, f1, f2 in pairs[:5]:
            self.txt.insert(tk.END, f"  {f1} <-> {f2}: {corr:.3f} (lag: {int(lag)})\n")
        
        self.txt.config(state=tk.DISABLED)
    
    def _exportCrossCorrMatrix(self):
        if self.crossCorrData is None:
            messagebox.showinfo("No data", "No cross-correlation data to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save cross-correlation matrix",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        with open(filename, 'w') as f:
            f.write(f"# Cross-Correlation Matrix\n")
            f.write(f"# Parameter: {self.crossCorrData['param'].upper()}\n")
            f.write(f"# Normalized: {self.crossCorrData['normalized']}\n")
            f.write(f"# Files: {', '.join(self.crossCorrData['file_names'])}\n")
            f.write("# Format: file1 file2 correlation lag_bins\n")
            f.write("#\n")
            
            corr_matrix = self.crossCorrData['corr_matrix']
            lag_matrix = self.crossCorrData['lag_matrix']
            for i, name_i in enumerate(self.crossCorrData['file_names']):
                for j, name_j in enumerate(self.crossCorrData['file_names']):
                    f.write(f"{name_i}\t{name_j}\t{corr_matrix[i, j]:.6f}\t{int(lag_matrix[i, j])}\n")
        
        messagebox.showinfo("Export complete", f"Matrix saved to {filename}")
    
    def _exportCrossCorrCSV(self):
        if self.crossCorrData is None:
            messagebox.showinfo("No data", "No cross-correlation data to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save cross-correlation matrix as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        df_corr = pd.DataFrame(self.crossCorrData['corr_matrix'], 
                               index=self.crossCorrData['file_names'],
                               columns=self.crossCorrData['file_names'])
        df_lag = pd.DataFrame(self.crossCorrData['lag_matrix'], 
                              index=self.crossCorrData['file_names'],
                              columns=self.crossCorrData['file_names'])
        
        with pd.ExcelWriter(filename.replace('.csv', '.xlsx'), engine='xlsxwriter') as writer:
            df_corr.to_excel(writer, sheet_name='Correlation')
            df_lag.to_excel(writer, sheet_name='Lag')
        
        df_corr.to_csv(filename)
        
        messagebox.showinfo("Export complete", f"Matrix saved to {filename}")

        
    def _align_signals(self, refs, frs):
        f0 = frs[0]
        out = []
        for s,f in zip(refs,frs):
            if len(f)==len(f0) and np.allclose(f,f0,rtol=1e-6): out.append(s); continue
            out.append(np.interp(f0, f, s))
        return np.array(out), f0

    def _l1_distance(self, a, b, norm):
        if norm:
            a=(a-np.mean(a))/(np.std(a)+1e-10)
            b=(b-np.mean(b))/(np.std(b)+1e-10)
        return np.mean(np.abs(a-b))

    def _l2_distance(self, a, b, norm):
        if norm:
            a=(a-np.mean(a))/(np.std(a)+1e-10)
            b=(b-np.mean(b))/(np.std(b)+1e-10)
        d=a-b
        return np.sqrt(np.mean(d*d))

    def _toggleLegend(self):
        if self.legendVisible.get():
            self.plotPane.add(self.legendFrame, weight=1)
            self.plotPaneT.add(self.legendFrameT, weight=1)
            self.plotPaneR.add(self.legendFrameR, weight=1)
        else:
            self.plotPane.forget(self.legendFrame)
            self.plotPaneT.forget(self.legendFrameT)
            self.plotPaneR.forget(self.legendFrameR)
    
    def _updateLegendPanel(self, legend_items, tab='freq'):
        if tab == 'freq':
            legendFrame = self.legendItemsFrame
            legendCanvas = self.legendCanvas
        elif tab == 'time':
            legendFrame = self.legendItemsFrameT
            legendCanvas = self.legendCanvasT
        elif tab == 'regex':
            legendFrame = self.legendItemsFrameR
            legendCanvas = self.legendCanvasR
        else:
            return
        
        for widget in legendFrame.winfo_children():
            widget.destroy()
        
        if not legend_items:
            ttk.Label(legendFrame, text="No data", foreground="gray").pack(pady=10)
            return
        
        for name, color in legend_items:
            frame = ttk.Frame(legendFrame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            colorLabel = tk.Label(frame, text="■", foreground=color, font=("", 12))
            colorLabel.pack(side=tk.LEFT, padx=(0, 5))
            
            displayName = name if len(name) <= 20 else name[:17] + "..."
            ttk.Label(frame, text=displayName, font=("", 9)).pack(side=tk.LEFT)
        
        legendCanvas.configure(scrollregion=legendCanvas.bbox("all"))

    def _updShapePlot(self):
        is_adaptive = self.shapeMetric.get() in ["al1", "al2"]
        for w in self.shapeBox.winfo_children():
            if isinstance(w, ttk.LabelFrame) and w['text'] == 'Adaptive Parameters':
                if is_adaptive:
                    w.pack(anchor="w", pady=(5,0), fill="x")
                else:
                    w.pack_forget()
        
        self.figSC.clear()
        if self.shapeShowActivity.get() and is_adaptive:
            self.axSC = self.figSC.add_subplot(211)
            self.axActivity = self.figSC.add_subplot(212)
        else:
            self.axSC = self.figSC.add_subplot(111)
        
        fmin=self.fmin.get(); fmax=self.fmax.get(); sstr=f"{fmin}-{fmax}ghz"
        param=self.shapeParam.get(); normalize=self.shapeNormalize.get(); metric=self.shapeMetric.get()
        sigs=[]; freqs=[]; names=[]
        for v,p,d in self.fls:
            if not v.get(): continue
            ext=Path(p).suffix.lower()
            if d.get('is_average', False) and d.get('custom_name'):
                lbl=d['custom_name']
            else:
                lbl = Path(p).stem if not p.startswith("<") else p[1:-1]
            if d.get('is_average', False) or ext in ['.s1p','.s2p','.s3p']:
                try:
                    ntw_full=d.get('ntwk_full'); cached=d.get('cached_range')
                    if ntw_full is None:
                        ntw_full=loadFile(p); d['ntwk_full']=ntw_full
                    if cached!=sstr:
                        ntw=ntw_full[sstr]; d['ntwk']=ntw; d['cached_range']=sstr
                    else:
                        ntw=d['ntwk']
                    s=getattr(ntw,param).s_db.flatten()
                    sigs.append(s); freqs.append(ntw.f); names.append(lbl)
                except: pass
        n=len(sigs)
        if n<2:
            self.axSC.text(0.5,0.5,"Need at least 2 files",ha="center",va="center",fontsize=12,color="gray")
            self.axSC.set_xticks([]); self.axSC.set_yticks([])
            self.cvSC.draw(); return
        sigs_aligned, fref = self._align_signals(sigs, freqs)
        
        weights = None
        if is_adaptive:
            derivs = self._compute_derivatives(sigs_aligned)
            activity = self._compute_activity_field(derivs, self.shapeAlpha.get(), self.shapeActivityType.get())
            activity_smooth = self._gaussian_smooth(activity, self.shapeSigma.get())
            weights = self._compute_adaptive_weights(activity_smooth, self.shapeGamma.get(), self.shapeWmin.get())
            
            if self.shapeShowActivity.get():
                self.axActivity.plot(fref, activity, 'b-', alpha=0.5, label='Raw activity')
                self.axActivity.plot(fref, activity_smooth, 'r-', label='Smoothed activity')
                self.axActivity.plot(fref, weights * np.max(activity_smooth) / np.max(weights), 'g-', label='Weights (scaled)')
                self.axActivity.set_xlabel('Frequency [Hz]')
                self.axActivity.set_ylabel('Activity / Weight')
                self.axActivity.legend()
                self.axActivity.grid(True, alpha=0.3)
        
        M=np.zeros((n,n)); L=np.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                if i==j:
                    if metric in ["xcorr","al1","al2"]: M[i,j]=1 if metric=="xcorr" else 0; L[i,j]=0
                    else: M[i,j]=0; L[i,j]=0
                    continue
                if metric=="xcorr":
                    max_lag_val = self.maxLag.get() if self.maxLag.get() > 0 else None
                    lag,val=self._cross_correlation(sigs_aligned[i],sigs_aligned[j],normalize, max_lag_val)
                    M[i,j]=M[j,i]=val; L[i,j]=L[j,i]=lag
                elif metric=="l1":
                    val=self._l1_distance(sigs_aligned[i],sigs_aligned[j],normalize)
                    M[i,j]=M[j,i]=val
                elif metric=="l2":
                    val=self._l2_distance(sigs_aligned[i],sigs_aligned[j],normalize)
                    M[i,j]=M[j,i]=val
                elif metric=="al1":
                    val=self._adaptive_distance(sigs_aligned[i],sigs_aligned[j],weights,1)
                    M[i,j]=M[j,i]=val
                elif metric=="al2":
                    val=self._adaptive_distance(sigs_aligned[i],sigs_aligned[j],weights,2)
                    M[i,j]=M[j,i]=val
        
        self.shapeData={'matrix':M,'lag':L,'names':names,'param':param,'normalized':normalize,'metric':metric}
        if weights is not None:
            self.shapeData['weights'] = weights
            self.shapeData['activity'] = activity_smooth
        
        if metric=="xcorr":
            im=self.axSC.imshow(M,cmap='RdBu_r',aspect='auto',interpolation='nearest',vmin=-1,vmax=1)
        else:
            vmax=np.percentile(M[np.triu_indices(n,1)],95) if n>1 else 1
            im=self.axSC.imshow(M,cmap='RdBu_r',aspect='auto',interpolation='nearest',vmin=0,vmax=vmax if vmax>0 else None)
        
        self.axSC.set_xticks(range(n)); self.axSC.set_yticks(range(n))
        self.axSC.set_xticklabels(names,rotation=45,ha='right'); self.axSC.set_yticklabels(names)
        for i in range(n):
            for j in range(n):
                if metric=="xcorr":
                    tc="white" if abs(M[i,j])>0.5 else "black"
                    txt=f'{M[i,j]:.2f}\n({int(L[i,j])})'
                else:
                    tc="white" if (M[i,j]>(np.max(M)*0.6 if np.max(M)>0 else 0)) else "black"
                    txt=f'{M[i,j]:.3f}'
                self.axSC.text(j,i,txt,ha="center",va="center",color=tc,fontsize=7)
        
        metric_names = {"xcorr":"Cross-Correlation", "l1":"L1 Distance", "l2":"L2 Distance", 
                       "al1":"Adaptive L1", "al2":"Adaptive L2"}
        t=f"{metric_names.get(metric,metric)} — {param.upper()}" + (" (Normalized)" if normalize else "")
        self.axSC.set_title(t)
        self.figSC.colorbar(im, ax=self.axSC, label=('corr' if metric=='xcorr' else 'distance'))
        self.figSC.tight_layout(); self.cvSC.draw()
        
        self.txt.config(state=tk.NORMAL); self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, "Shape Comparison\n")
        self.txt.insert(tk.END, f"Param: {param.upper()} | Metric: {metric} | Mode: {'Normalized' if normalize else 'Raw'}\n")
        if is_adaptive:
            self.txt.insert(tk.END, f"Adaptive params: α={self.shapeAlpha.get():.1f} σ={self.shapeSigma.get()} γ={self.shapeGamma.get():.1f} w_min={self.shapeWmin.get():.2f}\n")
            self.txt.insert(tk.END, f"Activity: {self.shapeActivityType.get()}\n")
        self.txt.insert(tk.END, f"Files: {n}\n")
        self.txt.insert(tk.END, "-"*40+"\n")
        vals=M[np.triu_indices(n,1)]
        if metric=="xcorr":
            self.txt.insert(tk.END, f"Max corr: {np.max(vals):.3f}\nMin corr: {np.min(vals):.3f}\nMean corr: {np.mean(vals):.3f}\n")
            lv=L[np.triu_indices(n,1)]
            self.txt.insert(tk.END, f"Mean lag: {np.mean(np.abs(lv)):.1f} bins\n\nTop pairs:\n")
            pairs=[]
            for i in range(n):
                for j in range(i+1,n):
                    pairs.append((M[i,j],L[i,j],names[i],names[j]))
            pairs.sort(reverse=True)
            for c,l,f1,f2 in pairs[:5]:
                self.txt.insert(tk.END, f"  {f1} <-> {f2}: {c:.3f} (lag {int(l)})\n")
        else:
            self.txt.insert(tk.END, f"Min dist: {np.min(vals):.4f}\nMax dist: {np.max(vals):.4f}\nMean dist: {np.mean(vals):.4f}\n\nClosest pairs:\n")
            pairs=[]
            for i in range(n):
                for j in range(i+1,n):
                    pairs.append((M[i,j],names[i],names[j]))
            pairs.sort()
            for d,f1,f2 in pairs[:5]:
                self.txt.insert(tk.END, f"  {f1} <-> {f2}: {d:.4f}\n")
        self.txt.config(state=tk.DISABLED)


    def _exportShapeMatrix(self):
        if self.shapeData is None:
            messagebox.showinfo("No data","No shape-comparison data to export"); return
        fn=filedialog.asksaveasfilename(title="Save matrix",defaultextension=".txt",filetypes=[("Text files","*.txt"),("All files","*.*")])
        if not fn: return
        with open(fn,'w') as f:
            f.write(f"# Shape Matrix\n# Metric: {self.shapeData['metric']}\n# Parameter: {self.shapeData['param'].upper()}\n# Normalized: {self.shapeData['normalized']}\n# Files: {', '.join(self.shapeData['names'])}\n# Format: file1 file2 value lag_bins\n#\n")
            M=self.shapeData['matrix']; L=self.shapeData['lag']; N=self.shapeData['names']
            for i,ni in enumerate(N):
                for j,nj in enumerate(N):
                    lb=int(L[i,j]) if self.shapeData['metric']=="xcorr" else 0
                    f.write(f"{ni}\t{nj}\t{M[i,j]:.6f}\t{lb}\n")
        messagebox.showinfo("Export complete", f"Matrix saved to {fn}")

    def _exportShapeCSV(self):
        if self.shapeData is None:
            messagebox.showinfo("No data","No shape-comparison data to export"); return
        fn=filedialog.asksaveasfilename(title="Save matrix as CSV",defaultextension=".csv",filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not fn: return
        names=self.shapeData['names']
        df=pd.DataFrame(self.shapeData['matrix'], index=names, columns=names)
        df.to_csv(fn)
        if self.shapeData['metric']=="xcorr":
            df_lag=pd.DataFrame(self.shapeData['lag'], index=names, columns=names)
            with pd.ExcelWriter(fn.replace('.csv','.xlsx'), engine='xlsxwriter') as w:
                df.to_excel(w, sheet_name='Value'); df_lag.to_excel(w, sheet_name='Lag')
        messagebox.showinfo("Export complete", f"Matrix saved to {fn}")


    def _updIntegPlot(self):
        self.figI.clear()
        self.axI = self.figI.add_subplot(111)
        
        fmin = self.fmin.get()
        fmax = self.fmax.get()
        sstr = f"{fmin}-{fmax}ghz"
        
        params_to_integrate = []
        if self.integS11.get(): params_to_integrate.append('s11')
        if self.integS12.get(): params_to_integrate.append('s12')
        if self.integS21.get(): params_to_integrate.append('s21')
        if self.integS22.get(): params_to_integrate.append('s22')
        
        if not params_to_integrate:
            self.axI.text(0.5, 0.5, "Select S-parameters to integrate", 
                         ha="center", va="center", fontsize=12, color="gray")
            self.axI.set_xticks([])
            self.axI.set_yticks([])
            self.cvI.draw()
            return
        
        scale_type = self.integScale.get()
        results = {}
        
        for v, p, d in self.fls:
            if not v.get():
                continue
            ext = Path(p).suffix.lower()
            if d.get('is_average', False) and d.get('custom_name'):
                lbl = d['custom_name']
            else:
                lbl = Path(p).stem if not p.startswith("<") else p[1:-1]
            if d.get('is_average', False) or ext in ['.s1p', '.s2p', '.s3p']:
                
                try:
                    ntw_full = d.get('ntwk_full')
                    cached_range = d.get('cached_range')
                    
                    if ntw_full is None:
                        ntw_full = loadFile(p)
                        d['ntwk_full'] = ntw_full
                    
                    if cached_range != sstr:
                        ntw = ntw_full[sstr]
                        d['ntwk'] = ntw
                        d['cached_range'] = sstr
                    else:
                        ntw = d['ntwk']
                    
                    if d.get('is_average', False) and d.get('custom_name'):
                        file_name = d['custom_name']
                    else:
                        file_name = Path(p).stem
                    if file_name not in results:
                        results[file_name] = {}
                    
                    freq = ntw.f

                    from scipy.interpolate import CubicSpline

                    for param in params_to_integrate:
                        s_data = getattr(ntw, param)
                        
                        if scale_type == 'db':
                            values = s_data.s_db.flatten()
                        else:
                            values = np.abs(s_data.s.flatten())

                        # freq_ghz = freq / 1e9
                        # cs = CubicSpline(freq_ghz, values)
                        # integral = cs.integrate(freq_ghz[0], freq_ghz[-1])

                        # x_norm = np.linspace(0, 1, len(values))
                        # cs = CubicSpline(x_norm, values)
                        # integral = cs.integrate(0, 1)

                        indices = np.arange(len(values))
                        cs = CubicSpline(indices, values)
                        integral = cs.integrate(indices[0], indices[-1])

                        results[file_name][param] = integral

                    # for param in params_to_integrate:
                    #    s_data = getattr(ntw, param)

                    #    if scale_type == 'db':
                    #       values = s_data.s_db.flatten()
                    #    else:
                    #       values = np.abs(s_data.s.flatten())

                    #   indices = np.arange(len(values))
                    #   integral = np.trapz(values, indices)

                        #results[file_name][param] = integral                                        
                except Exception as e:
                    pass
            
        if not results:
            self.axI.text(0.5, 0.5, "No valid data to integrate", 
                         ha="center", va="center", fontsize=12, color="gray")
            self.axI.set_xticks([])
            self.axI.set_yticks([])
            self.cvI.draw()
            return
        
        sort_by = self.integSortBy.get()
        ascending = self.integSortAsc.get()
        
        if sort_by == 'name':
            sorted_items = sorted(results.items(), key=lambda x: x[0], reverse=not ascending)
        elif sort_by == 'total':
            sorted_items = sorted(results.items(), 
                                key=lambda x: sum(x[1].values()), 
                                reverse=not ascending)
        else:
            sorted_items = sorted(results.items(), 
                                key=lambda x: x[1].get(sort_by, 0), 
                                reverse=not ascending)

        sorted_names = [item[0] for item in sorted_items]
        
        self.integData = results

        n_files = len(sorted_names)
        n_params = len(params_to_integrate)
        x = np.arange(n_files)
        width = 0.8 / n_params

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, param in enumerate(params_to_integrate):
            values = [results[fname].get(param, 0) for fname in sorted_names]
            self.axI.bar(x + i * width, values, width, label=param.upper(), color=colors[i])

        self.axI.set_xlabel('Files')
        self.axI.set_ylabel(f'Integrated value')

        sort_label = "name" if sort_by == "name" else f"{sort_by.upper()} {'↑' if ascending else '↓'}"
        self.axI.set_title(f'Integration of S-parameters ({scale_type} scale, sorted by {sort_label})')

        self.axI.set_xticks(x + width * (n_params - 1) / 2)
        self.axI.set_xticklabels(sorted_names, rotation=45, ha='right')
        self.axI.legend()
        self.axI.grid(True, alpha=0.3)
        
        self.figI.tight_layout()
        self.cvI.draw()
        
        self.txt.config(state=tk.NORMAL)
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, f"Integration Results\n")
        self.txt.insert(tk.END, f"Scale: {scale_type}\n")
        self.txt.insert(tk.END, f"Frequency range: {fmin}-{fmax} GHz\n")
        self.txt.insert(tk.END, f"Sort: {sort_label}\n")
        self.txt.insert(tk.END, "-" * 40 + "\n")
        
        for fname in sorted_names:
            params = results[fname]
            self.txt.insert(tk.END, f"\n{fname}:\n")
            total = sum(params.values())
            for param, value in params.items():
                self.txt.insert(tk.END, f"  {param.upper()}: {value:.3e}\n")
            if len(params) > 1:
                self.txt.insert(tk.END, f"  TOTAL: {total:.3e}\n")
        
        self.txt.config(state=tk.DISABLED)


    def _exportIntegResults(self):
        if self.integData is None:
            messagebox.showinfo("No data", "No integration data to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save integration results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        with open(filename, 'w') as f:
            f.write("# S-parameter Integration Results\n")
            f.write(f"# Scale: {self.integScale.get()}\n")
            f.write(f"# Frequency range: {self.fmin.get()}-{self.fmax.get()} GHz\n")
            f.write("# Format: filename parameter integral_value\n")
            f.write("#\n")
            
            for fname, params in self.integData.items():
                for param, value in params.items():
                    f.write(f"{fname}\t{param}\t{value:.6e}\n")
        
        messagebox.showinfo("Export complete", f"Results saved to {filename}")

    def _exportIntegCSV(self):
        if self.integData is None:
            messagebox.showinfo("No data", "No integration data to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save integration results as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        df = pd.DataFrame.from_dict(self.integData, orient='index')
        df.to_csv(filename)
        
        messagebox.showinfo("Export complete", f"Results saved to {filename}")


    def _showStyleMenu(self, event, filepath):
        file_data = None
        for v, p, d in self.fls:
            if p == filepath:
                file_data = d
                break
        
        if file_data is None:
            return
        
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Edit line style...", 
                        command=lambda: self._editLineStyle(filepath, file_data))
        
        menu.post(event.x_root, event.y_root)
    
    def _editLineStyle(self, filepath, file_data):
        dialog = tk.Toplevel(self)
        dialog.title(f"Edit style: {Path(filepath).name}")
        dialog.geometry("350x300")
        dialog.transient(self)
        dialog.grab_set()
        
        current_color = file_data.get('line_color')
        if current_color:
            try:
                rgb = matplotlib.colors.to_rgb(current_color)
                r, g, b = [int(x * 255) for x in rgb]
            except:
                r, g, b = 128, 128, 128
        else:
            r, g, b = 128, 128, 128
        
        ttk.Label(dialog, text="Color (RGB):", font=("", 10, "bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10,5), sticky="w")
        
        ttk.Label(dialog, text="Red:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        redVar = tk.IntVar(value=r)
        redSlider = ttk.Scale(dialog, from_=0, to=255, variable=redVar, orient=tk.HORIZONTAL, length=200)
        redSlider.grid(row=1, column=1, padx=10, pady=5)
        redValue = ttk.Label(dialog, text=str(r), width=4)
        redValue.grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(dialog, text="Green:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        greenVar = tk.IntVar(value=g)
        greenSlider = ttk.Scale(dialog, from_=0, to=255, variable=greenVar, orient=tk.HORIZONTAL, length=200)
        greenSlider.grid(row=2, column=1, padx=10, pady=5)
        greenValue = ttk.Label(dialog, text=str(g), width=4)
        greenValue.grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Label(dialog, text="Blue:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        blueVar = tk.IntVar(value=b)
        blueSlider = ttk.Scale(dialog, from_=0, to=255, variable=blueVar, orient=tk.HORIZONTAL, length=200)
        blueSlider.grid(row=3, column=1, padx=10, pady=5)
        blueValue = ttk.Label(dialog, text=str(b), width=4)
        blueValue.grid(row=3, column=2, padx=5, pady=5)
        
        ttk.Label(dialog, text="Line width:").grid(row=4, column=0, padx=10, pady=(15,5), sticky="w")
        widthVar = tk.DoubleVar(value=file_data.get('line_width', 1.0))
        widthSpin = ttk.Spinbox(dialog, from_=0.5, to=5.0, increment=0.5, textvariable=widthVar, width=10)
        widthSpin.grid(row=4, column=1, padx=10, pady=(15,5), sticky="w")
        
        ttk.Label(dialog, text="Preview:").grid(row=5, column=0, padx=10, pady=(15,5), sticky="w")
        previewFrame = ttk.Frame(dialog, relief=tk.SUNKEN, borderwidth=2)
        previewFrame.grid(row=5, column=1, padx=10, pady=(15,5), sticky="ew")
        previewLabel = tk.Label(previewFrame, text="━━━━━━━━━", font=("", 14), background="white")
        previewLabel.pack(padx=20, pady=10)
        
        def updatePreview(*args):
            r = redVar.get()
            g = greenVar.get()
            b = blueVar.get()
            redValue.config(text=str(r))
            greenValue.config(text=str(g))
            blueValue.config(text=str(b))
            
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            width = int(widthVar.get() * 2)
            previewLabel.config(foreground=hex_color, font=("", 10 + width))
        
        redVar.trace('w', updatePreview)
        greenVar.trace('w', updatePreview)
        blueVar.trace('w', updatePreview)
        widthVar.trace('w', updatePreview)
        updatePreview()
        
        buttonFrame = ttk.Frame(dialog)
        buttonFrame.grid(row=6, column=0, columnspan=3, pady=20)
        
        def apply():
            r = redVar.get()
            g = greenVar.get()
            b = blueVar.get()
            file_data['line_color'] = f'#{r:02x}{g:02x}{b:02x}'
            file_data['line_width'] = widthVar.get()
            dialog.destroy()
            self._updAll()
        
        def reset_default():
            file_data['line_color'] = None
            file_data['line_width'] = 1.0
            dialog.destroy()
            self._updAll()
        
        ttk.Button(buttonFrame, text="Apply", command=apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttonFrame, text="Reset to Default", command=reset_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttonFrame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

if __name__ == "__main__":
    App().mainloop()
