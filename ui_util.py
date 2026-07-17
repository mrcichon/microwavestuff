import tkinter as tk
from tkinter import ttk


def bind_enter(container, callback):
    # bind the Enter key on every entry/spinbox under container to callback
    for w in container.winfo_children():
        if isinstance(w, (ttk.Entry, ttk.Spinbox, tk.Entry, tk.Spinbox)):
            w.bind("<Return>", lambda e: callback())
        bind_enter(w, callback)
