# ui_main.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as mpatches

from ui_tab_freq import TabFreq
from ui_tab_time import TabTime
from ui_tab_regex import TabRegex
from ui_tab_overlap import TabOverlap
from ui_tab_variance import TabVariance
from ui_tab_shape import TabShapeComparison
from ui_tab_integrate import TabIntegrate
from ui_tab_td_analysis import TabTDAnalysis
from ui_tab_polar import TabPolar

try:
    from ui_tab_ml import TabMLTraining
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

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
            return True
        except (ValueError, TypeError):
            super().set(self._last_valid)
            return False


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("i can see through your skin")
        self.geometry("1400x800")
        
        self.fls = []
        self.fmin = ValidatedDoubleVar(value=0.4)
        self.fmax = ValidatedDoubleVar(value=4.0)
        self.avgCounter = 0
        
        self.current_tab = None
        self.mrk = []
        self.mtxt = []
        
        self.legendVisible = tk.BooleanVar(value=True)
        self.legendOnPlot = tk.BooleanVar(value=False)
        
        self._makeUi()
        self._updAll()
    
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
        
        ttk.Button(lfrm, text="Dodaj pliki", command=self._addFiles).pack(anchor="w", pady=(0,10))
        ttk.Button(lfrm, text="UsuÅ„ wszystkie markery", command=self._clearM).pack(anchor="w", pady=(0,10))
        ttk.Button(lfrm, text="UsuÅ„ zaznaczone pliki", command=self._deleteSelectedFiles).pack(anchor="w", pady=(0,10))
        ttk.Button(lfrm, text="Average files", command=self._avgFiles).pack(anchor="w", pady=(0,10))
        
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
        
        self.nb = ttk.Notebook(self)
        self.nb.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.nb.bind("<<NotebookTabChanged>>", self._onTab)
        
        self._create_freq_tab()
        self._create_time_tab()
        self._create_regex_tab()
        self._create_overlap_tab()
        self._create_variance_tab()
        self._create_shape_tab()
        self._create_integrate_tab()
        self._create_td_analysis_tab()
        if ML_AVAILABLE:
            self._create_ml_tab()
        self._create_polar_tab()
    
    def _create_freq_tab(self):
        frmF = ttk.Frame(self.nb)
        self.nb.add(frmF, text="Wykresy")
        
        self.plotPaneF = ttk.PanedWindow(frmF, orient=tk.HORIZONTAL)
        self.plotPaneF.pack(fill=tk.BOTH, expand=True)
        
        plotFrame = ttk.Frame(self.plotPaneF)
        self.plotPaneF.add(plotFrame, weight=3)
        
        self.figF = plt.figure(figsize=(10,7))
        self.cvF = FigureCanvasTkAgg(self.figF, master=plotFrame)
        canvas_widget = self.cvF.get_tk_widget()
        
        toolbar_frame = ttk.Frame(plotFrame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.tbF = NavigationToolbar2Tk(self.cvF, toolbar_frame)
        self.tbF.update()
        self.tbF.pack(side=tk.LEFT, fill=tk.X, expand=True)
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.legendFrameF = ttk.Frame(self.plotPaneF)
        self.plotPaneF.add(self.legendFrameF, weight=1)
        ttk.Label(self.legendFrameF, text="Legend", font=("", 10, "bold")).pack(pady=5)
        
        legendScrollF = ttk.Scrollbar(self.legendFrameF)
        legendScrollF.pack(side=tk.RIGHT, fill=tk.Y)
        self.legendCanvasF = tk.Canvas(self.legendFrameF, yscrollcommand=legendScrollF.set, width=150)
        self.legendCanvasF.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        legendScrollF.config(command=self.legendCanvasF.yview)
        
        self.legendItemsFrameF = ttk.Frame(self.legendCanvasF)
        self.legendCanvasF.create_window((0,0), window=self.legendItemsFrameF, anchor="nw")
        
        def updateLegendScrollF(event=None):
            self.legendCanvasF.configure(scrollregion=self.legendCanvasF.bbox("all"))
        self.legendItemsFrameF.bind("<Configure>", updateLegendScrollF)
        
        if not self.legendVisible.get():
            self.plotPaneF.forget(self.legendFrameF)
        
        self.tab_freq = TabFreq(
            parent=frmF,
            fig=self.figF,
            canvas=self.cvF,
            legend_frame=self.legendItemsFrameF,
            legend_canvas=self.legendCanvasF,
            get_files_func=self.get_files,
            get_freq_range_func=self.get_freq_range,
            get_legend_on_plot_func=lambda: self.legendOnPlot.get()
        )
        
        self.cvF.mpl_connect('button_press_event', self._onClick)
        self.cvF.mpl_connect('pick_event', self._onPick)
    
    def _create_time_tab(self):
        frmT = ttk.Frame(self.nb)
        self.nb.add(frmT, text="Time domain")
        
        self.plotPaneT = ttk.PanedWindow(frmT, orient=tk.HORIZONTAL)
        self.plotPaneT.pack(fill=tk.BOTH, expand=True)
        
        plotFrameT = ttk.Frame(self.plotPaneT)
        self.plotPaneT.add(plotFrameT, weight=3)
        
        self.figT = plt.figure(figsize=(10,10))
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
        
        if not self.legendVisible.get():
            self.plotPaneT.forget(self.legendFrameT)
        
        self.tab_time = TabTime(
            parent=frmT,
            fig=self.figT,
            canvas=self.cvT,
            legend_frame=self.legendItemsFrameT,
            legend_canvas=self.legendCanvasT,
            get_files_func=self.get_files,
            get_freq_range_func=self.get_freq_range,
            get_legend_on_plot_func=lambda: self.legendOnPlot.get()
        )
        
        self.cvT.mpl_connect('button_press_event', self._onClick)
        self.cvT.mpl_connect('pick_event', self._onPick)
    
    def _create_regex_tab(self):
        frmR = ttk.Frame(self.nb)
        self.nb.add(frmR, text="Regex Highlighting")
        
        self.plotPaneR = ttk.PanedWindow(frmR, orient=tk.HORIZONTAL)
        self.plotPaneR.pack(fill=tk.BOTH, expand=True)
        
        plotFrameR = ttk.Frame(self.plotPaneR)
        self.plotPaneR.add(plotFrameR, weight=3)
        
        self.figR = plt.figure(figsize=(10,7))
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
            self.plotPaneR.forget(self.legendFrameR)
        
        self.tab_regex = TabRegex(
            parent=frmR,
            fig=self.figR,
            canvas=self.cvR,
            legend_frame=self.legendItemsFrameR,
            legend_canvas=self.legendCanvasR,
            get_files_func=self.get_files,
            get_freq_range_func=self.get_freq_range,
            get_legend_on_plot_func=lambda: self.legendOnPlot.get()
        )
        
    def _create_overlap_tab(self):
        frmO = ttk.Frame(self.nb)
        self.nb.add(frmO, text="Range Overlaps")
        
        self.figO, self.axO = plt.subplots(figsize=(10,7))
        self.cvO = FigureCanvasTkAgg(self.figO, master=frmO)
        self.cvO.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.tbO = NavigationToolbar2Tk(self.cvO, frmO)
        self.tbO.update()
        self.tbO.pack(fill=tk.X)
        
        self.tab_overlap = TabOverlap(
            parent=frmO,
            fig=self.figO,
            ax=self.axO,
            canvas=self.cvO,
            get_files_func=self.get_files,
            get_freq_range_func=self.get_freq_range,
            get_regex_tab_func=lambda: self.tab_regex
        )

    def _create_variance_tab(self):
        frmV = ttk.Frame(self.nb)
        self.nb.add(frmV, text="Variance Analysis")
        
        self.figV, self.axV = plt.subplots(figsize=(10,8))
        self.cvV = FigureCanvasTkAgg(self.figV, master=frmV)
        self.cvV.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.tbV = NavigationToolbar2Tk(self.cvV, frmV)
        self.tbV.update()
        self.tbV.pack(fill=tk.X)
        
        self.tab_variance = TabVariance(
            parent=frmV,
            fig=self.figV,
            ax=self.axV,
            canvas=self.cvV,
            get_files_func=self.get_files,
            get_freq_range_func=self.get_freq_range
        )
        
        self.cvV.mpl_connect('button_press_event', self._onClickVar)
        self.cvV.mpl_connect('pick_event', self._onPickVar)
        self.cvV.mpl_connect('motion_notify_event', self._onMotionVar)
    
    def _create_shape_tab(self):
        frmSC = ttk.Frame(self.nb)
        self.nb.add(frmSC, text="Shape Comparison")
        
        self.figSC = plt.figure(figsize=(10,8))
        self.cvSC = FigureCanvasTkAgg(self.figSC, master=frmSC)
        self.cvSC.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.tbSC = NavigationToolbar2Tk(self.cvSC, frmSC)
        self.tbSC.update()
        self.tbSC.pack(fill=tk.X)
        
        self.tab_shape = TabShapeComparison(
            parent=frmSC,
            fig=self.figSC,
            canvas=self.cvSC,
            get_files_func=self.get_files,
            get_freq_range_func=self.get_freq_range
        )
    
    def _create_integrate_tab(self):
        frmI = ttk.Frame(self.nb)
        self.nb.add(frmI, text="Integration")
        
        self.figI, self.axI = plt.subplots(figsize=(10,8))
        self.cvI = FigureCanvasTkAgg(self.figI, master=frmI)
        self.cvI.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.tbI = NavigationToolbar2Tk(self.cvI, frmI)
        self.tbI.update()
        self.tbI.pack(fill=tk.X)
        
        self.tab_integrate = TabIntegrate(
            parent=frmI,
            fig=self.figI,
            ax=self.axI,
            canvas=self.cvI,
            get_files_func=self.get_files,
            get_freq_range_func=self.get_freq_range
        )
    
    def _create_td_analysis_tab(self):
        frmTDA = ttk.Frame(self.nb)
        self.nb.add(frmTDA, text="TD Analysis")
        
        self.figTDA = plt.figure(figsize=(10,8))
        self.cvTDA = FigureCanvasTkAgg(self.figTDA, master=frmTDA)
        self.cvTDA.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.tbTDA = NavigationToolbar2Tk(self.cvTDA, frmTDA)
        self.tbTDA.update()
        self.tbTDA.pack(fill=tk.X)
        
        self.tab_td_analysis = TabTDAnalysis(
            parent=frmTDA,
            fig=self.figTDA,
            canvas=self.cvTDA,
            get_files_func=self.get_files,
            get_freq_range_func=self.get_freq_range
        )
        
        self.cvTDA.mpl_connect('button_press_event', self._onClickTDA)
    
    def _create_ml_tab(self):
        frmM = ttk.Frame(self.nb)
        self.nb.add(frmM, text="Model Training")
        
        self.figM = plt.figure(figsize=(12,5))
        self.cvM = FigureCanvasTkAgg(self.figM, master=frmM)
        self.cvM.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.tbM = NavigationToolbar2Tk(self.cvM, frmM)
        self.tbM.update()
        self.tbM.pack(fill=tk.X)
        
        self.tab_ml = TabMLTraining(
            parent=frmM,
            fig=self.figM,
            canvas=self.cvM,
            get_files_func=self.get_files,
            get_freq_range_func=self.get_freq_range
        )
        
    def _create_polar_tab(self):
        frmPolar = ttk.Frame(self.nb)
        self.nb.add(frmPolar, text="Polar Plots")
        
        self.figPolar = plt.figure(figsize=(10, 8), constrained_layout=True)
        self.axPolar = self.figPolar.add_subplot(111, projection="polar")
        self.cvPolar = FigureCanvasTkAgg(self.figPolar, master=frmPolar)
        self.cvPolar.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.tbPolar = NavigationToolbar2Tk(self.cvPolar, frmPolar)
        self.tbPolar.update()
        self.tbPolar.pack(fill=tk.X)
        
        self.tab_polar = TabPolar(
            parent=frmPolar,
            fig=self.figPolar,
            ax=self.axPolar,
            canvas=self.cvPolar
        )



    def get_files(self):
        return self.fls
    
    def get_freq_range(self):
        fmin = self.fmin.get()
        fmax = self.fmax.get()
        sstr = f"{fmin}-{fmax}ghz"
        return (fmin, fmax, sstr)
    
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
                chk.bind("<Button-3>", lambda e, path=p: self._showStyleMenu(e, path))
                d = {'line_color': None, 'line_width': 1.0}
                self.fls.append((v, p, d))
        self.fileListCanvas.configure(scrollregion=self.fileListCanvas.bbox("all"))
        self._updAll()
    
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
                    name = d.get('custom_name', 'average')
                    chk = tk.Checkbutton(self.fbox, text=f"[AVG] {name}", variable=v, 
                                       command=self._updAll, fg="blue", activeforeground="blue")
                    chk.pack(anchor="w")
                else:
                    chk = ttk.Checkbutton(self.fbox, text=Path(p).name, variable=v, command=self._updAll)
                    chk.pack(anchor="w")
                chk.bind("<Button-3>", lambda e, path=p: self._showStyleMenu(e, path))
            
            self.fileListCanvas.configure(scrollregion=self.fileListCanvas.bbox("all"))
            self._updAll()
            messagebox.showinfo("Files removed", f"Removed {len(toDelete)} items from the list")
    
    def _avgFiles(self):
        dialog = tk.Toplevel(self)
        dialog.title("Average S-parameter files")
        dialog.geometry("450x650")
        dialog.transient(self)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select averaging mode:", font=("", 10, "bold")).pack(pady=10)
        
        mode = tk.StringVar(value="loaded")
        ttk.Radiobutton(dialog, text="Average loaded files", variable=mode, value="loaded").pack(anchor="w", padx=20)
        ttk.Radiobutton(dialog, text="Load and average new files", variable=mode, value="new").pack(anchor="w", padx=20)
        ttk.Radiobutton(dialog, text="Load folder structure (group by base name)", 
                       variable=mode, value="folder").pack(anchor="w", padx=20)
        
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
        
        regexFrame = ttk.LabelFrame(dialog, text="Folder mode: Grouping pattern")
        regexFrame.pack(fill="x", padx=10, pady=(10,5))
        
        ttk.Label(regexFrame, text="Regex to extract base name (must have 1 capture group):").pack(anchor="w", padx=5, pady=(5,2))
        regexVar = tk.StringVar(value=r'^(.+)_\d+$')
        regexEntry = ttk.Entry(regexFrame, textvariable=regexVar, width=50)
        regexEntry.pack(anchor="w", padx=5, pady=(0,5), fill="x")
        
        def process():
            if mode.get() == "loaded":
                selected = listbox.curselection()
                if len(selected) < 2:
                    messagebox.showwarning("Selection error", "Select at least 2 files to average")
                    return
                indices = [file_indices[s] for s in selected]
                self._performAverage(indices, nameVar.get())
                dialog.destroy()
            elif mode.get() == "new":
                dialog.destroy()
                self._loadAndAverage(nameVar.get())
            else:
                dialog.destroy()
                self._loadAndAverageFolderStructure(regexVar.get())
        
        buttonFrame = ttk.Frame(dialog)
        buttonFrame.pack(pady=10)
        ttk.Button(buttonFrame, text="OK", command=process).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttonFrame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
    
    def _performAverage(self, indices, name):
        from sparams_io import loadFile
        
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
        chk = tk.Checkbutton(self.fbox, text=f"[AVG] {name}", variable=v, command=self._updAll, 
                           fg="blue", activeforeground="blue")
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
        from sparams_io import loadFile
        
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
        chk = tk.Checkbutton(self.fbox, text=f"[AVG] {name}", variable=v, command=self._updAll,
                           fg="blue", activeforeground="blue")
        chk.pack(anchor="w")
        
        avg_path = f"<average_{self.avgCounter}>"
        d = {
            'ntwk_full': avg_network,
            'is_average': True,
            'source_files': list(files),
            'line_color': None,
            'line_width': 2.0,
            'custom_name': name
        }
        
        chk.bind("<Button-3>", lambda e: self._showStyleMenu(e, avg_path))
        self.fls.append((v, avg_path, d))
        
        self.fileListCanvas.configure(scrollregion=self.fileListCanvas.bbox("all"))
        self._updAll()
    
    def _loadAndAverageFolderStructure(self, grouping_regex):
        import re
        from sparams_io import loadFile
        
        parent_dir = filedialog.askdirectory(
            title="Select parent folder (each subfolder will be processed)"
        )
        
        if not parent_dir:
            return
        
        try:
            pattern = re.compile(grouping_regex)
        except Exception as e:
            messagebox.showerror("Invalid Regex", f"Regex pattern is invalid:\n{str(e)}")
            return
        
        try:
            subdirs = [d for d in os.listdir(parent_dir) 
                      if os.path.isdir(os.path.join(parent_dir, d))]
            
            if not subdirs:
                messagebox.showwarning("No subfolders", 
                                     f"No subfolders found in:\n{parent_dir}")
                return
            
            total_created = 0
            
            for subdir_name in sorted(subdirs):
                subdir_path = os.path.join(parent_dir, subdir_name)
                
                s_files = []
                for root, dirs, files in os.walk(subdir_path):
                    for file in files:
                        if file.lower().endswith(('.s1p', '.s2p', '.s3p')):
                            s_files.append(os.path.join(root, file))
                
                if not s_files:
                    continue
                
                groups = {}
                
                for filepath in s_files:
                    filename = Path(filepath).stem
                    match = pattern.match(filename)
                    
                    if match and len(match.groups()) >= 1:
                        base_name = match.group(1)
                        if base_name not in groups:
                            groups[base_name] = []
                        groups[base_name].append(filepath)
                
                for base_name, file_list in sorted(groups.items()):
                    if len(file_list) < 2:
                        continue
                    
                    networks = []
                    for filepath in file_list:
                        try:
                            ntw = loadFile(filepath)
                            networks.append(ntw)
                        except:
                            pass
                    
                    if len(networks) < 2:
                        continue
                    
                    avg_network = self._computeAverage(networks)
                    if avg_network is None:
                        continue
                    
                    self.avgCounter += 1
                    avg_name = f"{subdir_name}_{base_name}"
                    v = tk.BooleanVar(value=True)
                    chk = tk.Checkbutton(self.fbox, text=f"[AVG] {avg_name}", 
                                       variable=v, command=self._updAll, 
                                       fg="blue", activeforeground="blue")
                    chk.pack(anchor="w")
                    
                    avg_path = f"<average_{self.avgCounter}>"
                    d = {
                        'ntwk_full': avg_network,
                        'is_average': True,
                        'source_files': file_list,
                        'line_color': None,
                        'line_width': 2.0,
                        'custom_name': avg_name
                    }
                    
                    chk.bind("<Button-3>", lambda e, path=avg_path: self._showStyleMenu(e, path))
                    self.fls.append((v, avg_path, d))
                    total_created += 1
            
            self.fileListCanvas.configure(scrollregion=self.fileListCanvas.bbox("all"))
            self._updAll()
            
            messagebox.showinfo("Batch Averaging Complete", 
                              f"Created {total_created} average(s) from {len(subdirs)} subfolder(s)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load folder structure:\n{str(e)}")
    
    def _computeAverage(self, networks):
        import skrf as rf
        
        if not networks:
            return None
        
        try:
            ref_freqs = networks[0].f
            n_ports = networks[0].nports
            
            for ntw in networks[1:]:
                if ntw.nports != n_ports:
                    return None
                if len(ntw.f) != len(ref_freqs) or not np.allclose(ntw.f, ref_freqs, rtol=1e-6):
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
            
        except Exception:
            return None
    
    def _onTab(self, e):
        tabTxt = self.nb.tab(self.nb.select(), "text")
        self.current_tab = self._get_tab_by_name(tabTxt)
        
        if self.current_tab is not None:
            self.current_tab.update()
        
        self._update_text_panel()
    
    def _get_tab_by_name(self, name):
        tab_map = {
            "Wykresy": self.tab_freq,
            "Time domain": self.tab_time,
            "Regex Highlighting": self.tab_regex,
            "Range Overlaps": self.tab_overlap,
            "Variance Analysis": self.tab_variance,
            "Shape Comparison": self.tab_shape,
            "Integration": self.tab_integrate,
            "TD Analysis": self.tab_td_analysis,
            "Polar Plots": self.tab_polar,
        }
        if ML_AVAILABLE:
            tab_map["Model Training"] = self.tab_ml
        return tab_map.get(name)
    
    def _updAll(self):
        if self.current_tab is not None:
            self.current_tab.update()
        self._update_text_panel()
    
    def _update_text_panel(self):
        self.txt.config(state=tk.NORMAL)
        self.txt.delete(1.0, tk.END)
        
        if self.current_tab and hasattr(self.current_tab, 'get_text_output'):
            text = self.current_tab.get_text_output()
            if text:
                self.txt.insert(tk.END, text)
        
        if self.mtxt:
            self.txt.insert(tk.END, "\n" + "="*40 + "\n")
            self.txt.insert(tk.END, "MARKERS\n")
            self.txt.insert(tk.END, "-"*40 + "\n")
            for marker_text in self.mtxt:
                self.txt.insert(tk.END, marker_text)
        
        self.txt.config(state=tk.DISABLED)
    
    def _clearM(self):
        for m in self.mrk:
            try:
                m['marker'].remove()
                m['text'].remove()
            except:
                pass
        self.mrk.clear()
        self.mtxt.clear()
        self._update_text_panel()
        if self.current_tab and hasattr(self.current_tab, 'canvas'):
            self.current_tab.canvas.draw()
    
    def _onClick(self, ev):
        pass
    
    def _onPick(self, ev):
        pass
    
    def _onClickVar(self, ev):
        pass
    
    def _onPickVar(self, ev):
        pass
    
    def _onMotionVar(self, ev):
        pass
    
    def _onClickTDA(self, ev):
        pass
    
    def _toggleLegend(self):
        if self.legendVisible.get():
            self.plotPaneF.add(self.legendFrameF, weight=1)
            self.plotPaneT.add(self.legendFrameT, weight=1)
            self.plotPaneR.add(self.legendFrameR, weight=1)
        else:
            self.plotPaneF.forget(self.legendFrameF)
            self.plotPaneT.forget(self.legendFrameT)
            self.plotPaneR.forget(self.legendFrameR)
    
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
        menu.add_command(label="Delete", 
                        command=lambda: self._deleteLine(filepath))
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
        previewLabel = tk.Label(previewFrame, text="â”â”â”â”â”â”â”â”â”", font=("", 14), background="white")
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
    
    def _deleteLine(self, filepath):
        self.fls = [(v, p, d) for v, p, d in self.fls if p != filepath]
        
        for widget in self.fbox.winfo_children():
            widget.destroy()
        
        for v, p, d in self.fls:
            v.set(True)
            if d.get('is_average', False):
                name = d.get('custom_name', 'average')
                chk = tk.Checkbutton(self.fbox, text=f"[AVG] {name}", variable=v, 
                                   command=self._updAll, fg="blue", activeforeground="blue")
            else:
                chk = tk.Checkbutton(self.fbox, text=Path(p).name, variable=v, command=self._updAll)
            chk.pack(anchor="w")
            chk.bind("<Button-3>", lambda e, path=p: self._showStyleMenu(e, path))
        
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


if __name__ == "__main__":
    App().mainloop()
