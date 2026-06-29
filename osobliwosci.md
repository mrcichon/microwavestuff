# Osobliwości

Rzeczy, które wyglądają na pomyłkę albo zaskakują, a są celowe. Część to dług, o którym po
prostu warto wiedzieć.

## Dane per plik

Plik na liście to krotka `(BooleanVar, path, dict)`. W `dict` trzyma się cały stan tego
pliku: cache network (`ntwk_full`, `ntwk`, `cached_range`), styl linii, `overlay_params`,
dane uśrednienia. Nie ma stałego zestawu kluczy, różne zakładki dorzucają swoje w miarę
potrzeb. Wygodne, bo wszystko trzyma jeden słownik, ale co w nim siedzi, widać dopiero z
kodu, który go zapisuje.

Uśrednienie nie ma prawdziwej ścieżki, dostaje sztuczną `<average_N>`. Kod czytający pliki
patrzy na `is_average`, żeby nie iść po nią na dysk.

Zakres f to string `"0.4-4.0ghz"` i robi dwie rzeczy naraz: jest kluczem cięcia w skrf
(`ntwk_full[sstr]`) i kluczem cache (`d['cached_range']`). Jeden string, dwa zastosowania.

## Wczytywanie

`loadFile` nie parsuje plików z VNA wprost. Zapisuje znormalizowany plik tymczasowy (przecinki
dziesiętne na kropki), daje go skrf do sparsowania i kasuje. Prościej niż przepisywać parser
touchstone pod polskie przecinki.

`skrf_patch` podmienia `time_gate` i `delay` na wersje z scikit-rf 0.17. Nowsze skrf zmieniło
zachowanie, a bramkowanie w zakładkach liczy na stare, więc patch je przywraca.

## Markery

Stan markerów żyje w `App.marker_data` po `id(fig)`, nie po nazwie zakładki. Zakładki czyszczą
osie przy każdym `update()`, więc markery przepadłyby przy przerysowaniu. Dlatego
`_hook_marker_redraw` podmienia `canvas.draw` i `canvas.draw_idle`, żeby tuż przed rysowaniem
nanieść markery z powrotem. To jedyne miejsce, gdzie kod nadpisuje metody matplotliba.

## Zakładki spoza wzorca

Field Heatmap nie ma wpisu w `_get_tab_by_name`. Celowo: sama wczytuje swój CSV, nie ma
`get_text_output`, więc generyczna obsługa po tytule nie ma do czego się podłączyć.

Polar Plots i Field Heatmap w ogóle nie ruszają globalnej listy plików ani zakresu f,
wczytują własne pliki z osobnego okna.

Overlay (`ui_tab_overlay`) nie ma swoich checkboxów na parametry. Steruje nim `overlay_params`
ustawiane per plik prawym klikiem na liście.

Regex jest jedyną zakładką z dwoma modułami analizy (`analysis_regex` plus
`analysis_regex_stats`). Trzyma też zoom między przerysowaniami, grzebiąc w historii toolbara
matplotliba, żeby back/home wracały do poprzedniego widoku zamiast resetować.

TD Analysis (widok `ui_tab_td_analysis`) paruje z `analysis_td_peaks`, nie
`analysis_td_analysis`. Rozjazd nazw widok/analiza, tylko tutaj.

Atrybut z ostatnim wynikiem nazywa się inaczej w każdej zakładce: `last_result`, `last_data`,
`shape_data`, `td_analysis_data`. Z zewnątrz nie przeszkadza, bo wszystko idzie przez
`get_text_output()`, ale przy czytaniu kodu zaskakuje.

## Dług i niepodpięty kod

Zakładka diff (`ui_tab_diff` + `analysis_diff`) jest kompletna i ma testy, ale nie jest
podpięta: `ui_main` jej nie importuje ani nie buduje. Albo niedokończona robota, albo
porzucona. Na razie zostaje tak.

Część metod importuje lokalnie (`from sparams_io import loadFile` w środku metody) zamiast na
górze pliku. Historyczne, nie ma w tym zamysłu.
