import skrf_patch  # noqa: F401  installs the skrf time_gate/delay patch on import

if __name__ == "__main__":
    from ui_main import App
    App().mainloop()
