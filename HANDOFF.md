# Handoff notes

Per-branch summaries for the merge session. Each section is self-contained.

## claude/moose-script-structures-dn2f74

**Ziel:** Ein Moose-Skript (C#) bauen, das die 13 Strukturen aus `benchmark/structures.py` in Moose nachbaut, über einen Ordnungs-Sweep löst und R/T/A plus Timings ausgibt, damit man das nicht mehr manuell durch die RCWA-Dialoge klicken muss. Dazu ein Python-Merge-Skript, das die Moose-CSV-Ausgabe nach `benchmark/moose_reference.json` einpflegt (für `plot_moose.py`).

**Was hat funktioniert:**
- Die MOOSE-API wurde aus der mitgelieferten Doxygen-`.qch`-Hilfedatei extrahiert (SQLite-DB mit zlib-komprimiertem HTML) und daraus C#-Stub-Klassen gebaut, gegen die sich das Skript mit `mcs` offline kompilieren und typprüfen lässt, ohne Moose selbst zu haben.
- Struktur-Übersetzung (Layer/Atom/GratingStructure-Aufbau, Materialien, Duty-Cycle-Gratings) wurde gegen echte Moose-Läufe verifiziert: 1D-Fälle (Gruppen A, B, D1) reproduzieren die alten manuellen Moose-Werte auf alle sechs Stellen exakt.
- Eine Energiebilanz-Prüfung (R+T+A muss 1 ergeben, Toleranz ~1e-6) hat sich als der zentrale Korrektheits-Wächter erwiesen — sie hat zwei der drei unten genannten Bugs selbst aufgedeckt, bevor sie durch Nachrechnen bestätigt wurden.
- Parallelisierung über einen Thread-Pool (mehrere Strukturen gleichzeitig lösen, je eigene `Rcwa`-Instanz) funktioniert und wurde mit einem Selbsttest (sequenziell vs. parallel, mehrfach wiederholt, bitweiser Vergleich) abgesichert; 3.7x Speedup auf 4 Kernen gemessen, Ergebnisse bit-identisch.
- Cross-Checks der Moose-Ergebnisse gegen grcwa (Laurent/Pol) und Ikarus (Li/NV) bei konvergierten Ordnungen zeigen für die 2D-Fälle, dass Moose nahe an den gut-faktorisierten Methoden (Pol/Li/NV) liegt, nicht an Laurent — das ist das erwartete Verhalten für eine ausgereifte RCWA-Implementierung und wurde als Plausibilitätsbeleg genutzt.

**Was war fehlerhaft / nicht weiter verfolgen:**
- Erste Skriptversion hatte drei Bugs, die zu falschen R/T-Werten führten (alle behoben, nicht wiederholen):
  1. `GetEfficiencyForGivenOrder`/`GetAbsorption` liefern **Prozent**, nicht Bruchteile — undokumentiert. Erkennbar daran, dass R+T+A konstant 100 statt 1 ergab.
  2. Der dritte Parameter des zirkulären `Atom`-Konstruktors (`Atom(x, y, r, mat)`) ist der **Radius**, nicht der Durchmesser. Der mitgelieferte Unit-Test (`TestAtom.TestCircular`) legt über seine Bounding-Box-Getter (`GetStartX`/`GetStopX`) nahe, es sei der Durchmesser — das ist irreführend, die Getter stimmen nicht mit dem überein, was der Solver tatsächlich rastert. Führte zu einem um Faktor 2 zu großen Kreis (Fall D2), R fiel von ~0.95 auf ~0.027. Wichtig: das ist **nicht** über die Seitenansicht (`ConvertToCaModel`/`SHOW_STRUCTURES`) erkennbar, da eine Seitenansicht die 2D-Draufsicht-Geometrie nicht zeigt — man muss die Zahlen prüfen, nicht das Bild.
  3. `GetEfficiencyForGivenOrder(..., rOutputPolarization)`: der Default `"in"` liefert nur den ko-polarisierten Anteil. Bei 2D-Gittern konvertieren schräge Ordnungen (beide Indizes ≠ 0) die Polarisation, ihr kreuzpolarisierter Anteil fehlte dann in der Summe. Betraf nur den einen 2D-Fall mit propagierenden schrägen Ordnungen. Lösung: Summe wird jetzt auf drei Arten gebildet (`"TE"+"TM"`, `"both"`, `"in"`) und die energieerhaltende Variante automatisch gewählt. Nebenbefund: `"both"` liefert entgegen der Doku-Vermutung nicht die Summe beider Polarisationen, sondern 0 — also auch keine gültige Wahl.
- Kein inhaltlicher Grund, an der Grundarchitektur (Stage-basierter Sweep, CSV-Zeile pro (Struktur, Ordnung), Energiebilanz als Gate) etwas zu ändern — die hat sich bewährt.

**Wo wir stehen geblieben sind:**
- Unmittelbar nächster Schritt: Der Nutzer wollte als Nächstes einen Sweep mit den exakten q-Werten aus der Python-Seite (`FULL_Q_LIST = 1,3,5,...,61` mit `GRCWA_NG1D_FROM_Q2D=1`) fahren, um Moose-Werte direkt mit den grcwa/Ikarus-Konvergenzkurven vergleichbar zu machen. Die passenden `SWEEP_1D`/`SWEEP_2D`-Arrays (als Moose-Maximalordnung `m = (q-1)/2` bzw. `m = (nG-1)/2`) sind im Skript bereits als auskommentierte Alternative hinterlegt, aber noch nicht real durchgerechnet/gemergt. Hinweis: die 1D-Liste enthält auch `q²`-Werte bis nG=3721, das ist ein ~7400×7400-Eigenwertproblem pro Punkt — teuer.
- Übergeordnete Richtung (noch nicht begonnen): mit Parallelisierung + höherem `PARALLEL_TASKS`/`PARALLEL_NG_LIMIT` einen vollständigen, konvergierten 2D-Sweep bis zu höheren Ordnungen (z.B. bis q=61) fahren, dessen Ergebnisse mergen und dann `plot_moose.py` als endgültigen Cross-Code-Vergleich (grcwa/Ikarus/Moose) über alle 13 Strukturen laufen lassen. Vor jedem größeren parallelen Lauf sollte `PARALLEL_SELFTEST` einmal auf der Zielmaschine bestätigt werden, da Thread-Sicherheit der `Rcwa`-Klasse nicht offiziell dokumentiert ist.
- Kein offener Geometrie-Rasterisierungs-Konflikt (FFT-Grid/Kantenpixel/effektives Medium) zwischen den Solvern wurde in diesem Chat untersucht — die drei gefundenen Diskrepanzen waren reine Moose-API-Missverständnisse (Skalierung, Konstruktor-Semantik, Polarisations-Default), keine numerischen/Rasterisierungs-Fragen.
