# Architektura

`microwavestuff` to aplikacja okienkowa (tkinter + matplotlib) do oglądania i analizy
S-parametrów oraz charakterystyk antenowych. Jeden plik wejściowy, reszta to zakładki,
każda od innego typu wykresu albo analizy.

## Uruchomienie

Wejście to `mainscript.py`: import `skrf_patch` (dla side effect, opis niżej) i
`App().mainloop()`. Działa też `python ui_main.py`, bo `ui_main.py` ma własny `__main__` i
też importuje `skrf_patch` na górze. Bramkowanie w zakładce regex zależy od `skrf_patch`,
więc odpalaj z jednego z tych wejść, nie z samej zakładki.

## App

`ui_main.App` (dziedziczy po `tk.Tk`) trzyma globalny stan i buduje UI. Po lewej panel
sterowania (zakres f, lista plików, przyciski, pole z wynikami), po prawej `ttk.Notebook`
z zakładkami.

Stan widoczny dla zakładek:
- `self.fls`: lista plików (model niżej),
- `self.fmin` / `self.fmax`: zakres f w GHz (`ValidatedDoubleVar`, domyślnie 0.4 i 4.0),
- `self.use_db_scale`, `self.legendOnPlot`, `self.legendVisible`, `self.markers_enabled`: przełączniki.

Zakładki nie odwołują się do `App` bezpośrednio. Stan dostają przez callbacki: `get_files`,
`get_freq_range`, `get_scale_mode` i `lambda: self.legendOnPlot.get()`. `get_freq_range()`
zwraca `(fmin, fmax, "0.4-4.0ghz")`. Ten string jest naraz kluczem cięcia w skrf i kluczem
cache.

## Model listy plików

Wpis w `self.fls` to krotka `(BooleanVar, path, dict)`:
- `BooleanVar`: zaznaczenie pliku (checkbox na liście),
- `path`: ścieżka, albo sztuczny `<average_N>` dla uśrednień,
- `dict`: stan per plik. Cache (`ntwk_full`, `ntwk`, `cached_range`), styl linii
  (`line_color`, `line_width`), `overlay_params`, a dla uśrednień `is_average`,
  `custom_name`, `source_files`.

Słownik jest współdzielony. Jak jedna zakładka wczyta i potnie network, zapis zostaje w
słowniku i następna bierze gotowe.

## Wczytywanie i cache

`sparams_io.get_cached_network(p, d, freq_range_str)` to jedyne wejście do wczytywania w
zakładkach. Wczytuje plik raz do `d['ntwk_full']`, tnie do zakresu, trzyma wynik w
`d['ntwk']` pod kluczem `d['cached_range']`. Dla pliku nie do użycia (złe rozszerzenie i nie
uśrednienie) albo gdy wczytanie padnie, zwraca `None` i loguje powód na stderr.

`sparams_io.loadFile(p)` ogarnia pliki prosto z VNA: kilka kodowań, rozpoznanie CSV z
analizatora widma, przecinki dziesiętne na kropki, zwrot `rf.Network`. Przecinki normalizuje
przez zapis pliku tymczasowego, który skrf parsuje, potem go kasuje.

## skrf_patch

`skrf_patch.py` przywraca `time_gate` i `delay` z scikit-rf 0.17. Nowsze skrf zmieniło te
metody, a zakładki z bramkowaniem liczą na stare. Import działa przez side effect: podpina
metody do `rf.Network`, więc nakłada się przy każdym imporcie. Dlatego oba wejścia importują
go pierwszego.

## Zakładki

Zakładka to klasa budowana w `App._create_*_tab()` i rejestrowana w notebooku przez
`nb.add(frame, text="...")`. Kontrakt: `update()` (przelicz i narysuj) oraz
`get_text_output()` (tekst do panelu po lewej).

Zmiana zakładki uruchamia `_onTab`: po tytule znajduje obiekt w `_get_tab_by_name`, odpala
`update()`, odświeża panel. `_updAll` (po zmianie plików albo zakresu) robi to samo dla
bieżącej zakładki.

Wykaz zakładek (tytuł, widok, analiza):
- Wykresy: `ui_tab_freq` / `analysis_freq`. Magnituda wybranych S-parametrów względem f.
- Time domain: `ui_tab_time` / `analysis_time`. Dziedzina czasu jednego parametru, z bramkowaniem.
- Regex Highlighting: `ui_tab_regex` / `analysis_regex` plus `analysis_regex_stats`. Sortuje
  krzywe po liczbie z nazwy pliku i zaznacza pasma z kilku analiz. Najcięższa zakładka, ma
  osobny `regex_tab.md`.
- Range Overlaps: `ui_tab_overlap` / `analysis_overlap`. Nakładanie się zakresów f.
- Shape Comparison: `ui_tab_shape` / `analysis_shape`. Macierz podobieństwa kształtu krzywych.
- TD Analysis: `ui_tab_td_analysis` / `analysis_td_peaks`. Piki w dziedzinie czasu dla S11 i S21.
- Polar Plots: `ui_tab_polar` / `analysis_polar`. Charakterystyki antenowe w układzie biegunowym.
- S-Param Overlay: `ui_tab_overlay` / `analysis_overlay`. Nakładka wybranych parametrów per plik.
- Field Heatmap: `ui_tab_field` / `analysis_field`. Heatmapa pola z pliku CSV.

Polar Plots i Field Heatmap nie ruszają globalnej listy plików ani zakresu, wczytują swoje
pliki z osobnego okna. Field Heatmap nie ma wpisu w `_get_tab_by_name` (celowo, powód w
`osobliwosci.md`).

## Warstwy

Na zakładkę dwie warstwy:
- `analysis_X.py`: czyste funkcje, bez tkinter. Wejście to dane, wyjście liczby albo
  słowniki. Testowalne bez ekranu.
- `ui_tab_X.py`: widok. Kontrolki, wywołania `analysis_X`, rysowanie, formatowanie tekstu.

`sparams_io.py`: wczytywanie i parsery formatów. `ui_util.py`: `bind_enter` (Enter w polu
przerysowuje). `conftest.py`: wspólne fikstury testów.

## Markery

Marker to przypięty punkt na wykresie z opisem (f albo czas, plus wartość). Stan trzyma
`App.marker_data`, słownik po `id(fig)`. Zakładki czyszczą osie w `update()`, więc markery
przepadłyby przy przerysowaniu. Dlatego `_hook_marker_redraw` podmienia `canvas.draw` i
`canvas.draw_idle`, żeby przed właściwym rysowaniem nanieść je znowu. Lewy klik wpina marker
w najbliższy punkt krzywej, prawy daje menu (usuń, dodaj hurtem). Szczegóły w `osobliwosci.md`.

## Testy

Testy w `tests/`, fikstury w `conftest.py`. Funkcje analizy sprawdza się wprost na sztucznym
network z fikstur. Zakładki sprawdza `test_tab_render.py`: składa każdą klasę z ręcznych
widgetów i callbacków z lambd, odpala `update()`, patrzy czy pusta lista nic nie rysuje, a
rodzina plików rysuje. Bez ekranu (brak X) testy zakładek się pomijają, nie wywalają. Pełny
przebieg to teraz 57 testów; jak odpalać, jest w `running_tests.txt`.
