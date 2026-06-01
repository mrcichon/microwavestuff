"""Headless smoke for the live tabs: construct each, run update() on empty input and
on the real file family, and check data actually reaches the plot. Needs a display
(skips otherwise). These guard the load/slice/cache wiring the cleanup refactors."""
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import pytest
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui_tab_freq import TabFreq
from ui_tab_time import TabTime
from ui_tab_regex import TabRegex
from ui_tab_integrate import TabIntegrate
from ui_tab_variance import TabVariance
from ui_tab_shape import TabShapeComparison
from ui_tab_td_analysis import TabTDAnalysis
from ui_tab_rozpierdol import TabRozpierdol
from ui_tab_overlap import TabOverlap
from ui_tab_polar import TabPolar
from ui_tab_field import TabField


def _cb(files):
    return {
        "files": lambda: files,
        "freq": lambda: (0.8, 2.0, "0.8-2.0ghz"),
        "legend": lambda: False,
        "scale": lambda: True,
        "regex": lambda: None,
    }


def _w(tk_root, ax=False, legend=False):
    parent = ttk.Frame(tk_root)
    control = ttk.Frame(parent)
    fig = plt.figure()
    canvas = FigureCanvasTkAgg(fig, master=parent)
    parts = [parent, control, fig]
    if ax:
        parts.append(fig.add_subplot(111))
    parts.append(canvas)
    if legend:
        parts += [ttk.Frame(parent), tk.Canvas(parent)]
    return parts, fig


def _build(name, tk_root, files):
    c = _cb(files)
    if name == "freq":
        p, fig = _w(tk_root, legend=True)
        return TabFreq(*p, c["files"], c["freq"], c["legend"], c["scale"]), fig
    if name == "time":
        p, fig = _w(tk_root, legend=True)
        return TabTime(*p, c["files"], c["freq"], c["legend"], c["scale"]), fig
    if name == "regex":
        p, fig = _w(tk_root, legend=True)
        return TabRegex(*p, c["files"], c["freq"], c["legend"], c["scale"]), fig
    if name == "overlay":
        p, fig = _w(tk_root, legend=True)
        return TabRozpierdol(*p, c["files"], c["freq"], c["legend"]), fig
    if name == "integrate":
        p, fig = _w(tk_root, ax=True)
        return TabIntegrate(*p, c["files"], c["freq"], c["scale"]), fig
    if name == "variance":
        p, fig = _w(tk_root, ax=True)
        return TabVariance(*p, c["files"], c["freq"]), fig
    if name == "shape":
        p, fig = _w(tk_root)
        return TabShapeComparison(*p, c["files"], c["freq"], c["scale"]), fig
    if name == "td":
        p, fig = _w(tk_root)
        return TabTDAnalysis(*p, c["files"], c["freq"]), fig
    if name == "overlap":
        p, fig = _w(tk_root, ax=True)
        return TabOverlap(*p, c["files"], c["freq"], c["regex"]), fig
    if name == "polar":
        p, fig = _w(tk_root, ax=True)
        return TabPolar(*p), fig
    if name == "field":
        p, fig = _w(tk_root, ax=True)
        return TabField(*p), fig
    raise KeyError(name)


ALL_TABS = ["freq", "time", "regex", "overlay", "integrate", "variance",
            "shape", "td", "overlap", "polar", "field"]
# tabs that load the S-param file family through the shared caching path
DATA_TABS = ["freq", "time", "regex", "overlay", "integrate", "variance", "shape", "td"]


def _prep(name, tab, files):
    # set the minimum state a tab needs to actually plot, so the data test exercises
    # the real path instead of a default-off no-op
    if name == "freq":
        tab.s11_var.set(True)
    elif name == "overlay":
        for _v, _p, d in files:
            d["overlay_params"] = {"s11"}


def _drew_something(fig):
    return any(ax.lines or ax.images or ax.patches or ax.collections for ax in fig.axes)


@pytest.mark.parametrize("name", ALL_TABS)
def test_tab_update_empty_does_not_raise_or_fabricate(tk_root, name):
    # the no-files state (e.g. a tab just switched to) must not raise and must not
    # invent data on the canvas
    tab, fig = _build(name, tk_root, [])
    tab.update()
    assert not _drew_something(fig), f"{name} drew artifacts from empty input"


@pytest.mark.parametrize("name", DATA_TABS)
def test_tab_plots_family_data(tk_root, name, family_files):
    tab, fig = _build(name, tk_root, family_files)
    _prep(name, tab, family_files)
    tab.update()
    # the family must actually reach the canvas, not get swallowed by a silent except
    assert _drew_something(fig), f"{name}: update() produced nothing from the file family"
