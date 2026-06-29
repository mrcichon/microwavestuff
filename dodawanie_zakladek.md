# Dodawanie zakładki

Zakładka to dwa pliki plus rejestracja w `ui_main`. Najszybciej: skopiuj zakładkę najbliższą
temu, co chcesz (np. `ui_tab_shape` albo `ui_tab_td_analysis` na prosty wykres, `ui_tab_freq`
jak potrzebujesz legendy) i przerób.

## 1. analysis_twoje.py

Czyste funkcje, bez tkinter. Liczenie osobno, formatowanie tekstu osobno:

```python
import numpy as np

def compute_twoje(files_data, param):
    ...
    return {"wynik": ..., "param": param}

def format_twoje_text(result):
    lines = ["Twoja analiza", "-" * 40]
    ...
    return "\n".join(lines)
```

## 2. ui_tab_twoje.py

Widok. Konstruktor bierze callbacki, nie `App`. Standardowy zestaw (jak we `freq`, `time`,
`regex`):

```python
class TabTwoje:
    def __init__(self, parent, control_frame, fig, canvas,
                 legend_frame, legend_canvas,
                 get_files_func, get_freq_range_func,
                 get_legend_on_plot_func, get_scale_mode_func):
        self.fig = fig
        self.canvas = canvas
        self.get_files = get_files_func
        self.get_freq_range = get_freq_range_func
        self.last_result = None
        self._build_ui(control_frame)
```

Prostsza zakładka (bez legendy) bierze mniej, patrz `ui_tab_shape` albo `ui_tab_td_analysis`.

Dane wczytuj przez wspólny cache:

```python
from sparams_io import get_cached_network, display_name

def _get_files_data(self):
    fmin, fmax, sstr = self.get_freq_range()
    out = []
    for v, p, d in self.get_files():
        if not v.get():
            continue
        ntw = get_cached_network(p, d, sstr)
        if ntw is None:
            continue
        out.append({"name": display_name(p, d), "ntw": ntw})
    return out
```

`update()` liczy i rysuje, `get_text_output()` zwraca tekst:

```python
def update(self):
    data = self._get_files_data()
    self.fig.clear()
    ax = self.fig.add_subplot(111)
    if not data:
        ax.text(0.5, 0.5, "Brak plików", ha="center", va="center", color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw()
        self.last_result = None
        return
    self.last_result = compute_twoje(data, ...)
    # rysowanie
    self.canvas.draw()

def get_text_output(self):
    return "" if self.last_result is None else format_twoje_text(self.last_result)
```

Pustą listę obsłuż wprost (nic nie rysuj), bo test pustego wejścia tego pilnuje.

## 3. Rejestracja w ui_main.py

Import na górze:

```python
from ui_tab_twoje import TabTwoje
```

Metoda budująca, najprościej skopiowana z istniejącej `_create_*_tab` i przerobiona. Składa
frame, `control_frame` na dole, figurę, canvas, toolbar, na końcu tworzy obiekt zakładki:

```python
def _create_twoje_tab(self):
    frm = ttk.Frame(self.nb)
    self.nb.add(frm, text="Twoja zakładka")
    control = ttk.Frame(frm); control.pack(side=tk.BOTTOM, fill=tk.X)
    self.figTwoje = plt.figure(figsize=(10, 8))
    self.cvTwoje = FigureCanvasTkAgg(self.figTwoje, master=frm)
    self.cvTwoje.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    NavigationToolbar2Tk(self.cvTwoje, frm).update()
    self.tab_twoje = TabTwoje(parent=frm, control_frame=control,
                              fig=self.figTwoje, canvas=self.cvTwoje,
                              get_files_func=self.get_files,
                              get_freq_range_func=self.get_freq_range)
```

Odpal ją w `_makeUi` razem z resztą `_create_*_tab`. Na koniec dopisz tytuł do
`_get_tab_by_name`, inaczej zakładka się pokaże, ale nie dostanie `update()` po zmianie plików
ani panelu tekstowego:

```python
"Twoja zakładka": self.tab_twoje,
```

## 4. Test

Dopisz zakładkę do `test_tab_render.py`: do `ALL_TABS` (test pustego wejścia), a jak rysuje
dane z plików, też do `DATA_TABS` (test rysowania rodziny) i do `_build`. Moduł analizy
przetestuj wprost w `tests/test_analysis_twoje.py` na fikstrach z `conftest.py`. Odpal
`python -m pytest tests/ -q`.

## Parsery formatów

Jak zakładka czyta inny format niż touchstone, parsery są w `sparams_io.py`: `parse_polar_rms`,
`parse_polar_pustelnik`, `parse_theta_phi_file`. Touchstone i CSV z analizatora widma ogarnia
`loadFile`.
