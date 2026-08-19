# Handoff notes

Per-branch notes for a later merge session. Each section is self-contained;
append new branches below, never edit or remove existing ones.

## claude/2d-structures-convergence-njqie2

**1. Ziel.** Bei den 2D-Strukturen des Benchmarks (`C1_Si_pillars`,
`C1b_Si_pillars_diffract`, `C2_Au_holes`, `D2_ikarus_cylinder_TE`) konvergiert
Moose offensichtlich gegen einen anderen Grenzwert als grcwa/Ikarus, während
alle 1D-Strukturen übereinstimmen. Aufgabe war, herauszufinden woran das liegt
— insbesondere ob es an unterschiedlicher Rasterisierung, n/λ-Konventionen
oder anderen Aufbau-Unterschieden zwischen den drei Plattformen liegt.

**2. Was hat funktioniert.**

- Ursache gefunden und von beiden Seiten (Python und Moose, per neu
  geschriebenem `benchmark/moose/moose_geometry_probe.cs`) verifiziert: die
  **2D-Rasterisierung in `structures.py`** (`layer_mask`, Rechteck-Zweig) trifft
  die nominale Pfeiler-/Lochbreite bei `NX_2D = 256` nicht exakt (linke
  Zellkante, striktes `<`; `0.6*256=153.6` etc. sind keine ganzen Zahlen). Diese
  eine Maske wird von grcwa UND Ikarus geteilt (deshalb stimmen die beiden
  Python-Codes trotzdem überein), aber sie weicht von der nominalen Geometrie
  ab, die Moose stattdessen direkt aus den Parametern baut.
- Kausalität sauber isoliert: bei exakt gleicher Ordnung/Code/Regel bewegt allein
  der Wechsel von Masken- auf Nominalbreite den Wert um 81–96 % in Richtung
  Moose, auf allen drei betroffenen Rechteck-Fällen. Kontrollrechnung mit zwei
  verschieden feinen exakten FFT-Gittern (260² vs. 1280²) zeigt, dass NICHT die
  Gitterauflösung/Aliasing der Effekt ist, sondern einzig die dargestellte
  Breite.
- Zusätzlich auf der Moose-Seite gefunden: `rRefinementFactorEpsFT` wird von
  Moose intern auf einen festen Bereich geklemmt (die UI/API akzeptiert nur
  Werte in diesem Bereich, kleinere werden stillschweigend angehoben) — das
  macht den ursprünglich im Skript verwendeten adaptiven Verfeinerungsmodus
  (`FFT_MODE=1`, gedacht um mit grcwas festem 256er-Gitter mitzuhalten) für den
  gesamten 2D-Sweep wirkungslos, weil der berechnete Wert unter der Untergrenze
  lag. Dadurch lief effektiv jeder 2D-Punkt mit dem Minimalwert statt mit
  wachsender Verfeinerung. Nach Angleichen der Konstanten an den tatsächlichen
  Moose-Bereich und Extrapolation gegen unendliche Verfeinerung verschwindet
  praktisch der gesamte Restunterschied zu Python (~1e-4).
- `D2_ikarus_cylinder_TE` (Kreis) ist NICHT durch dieselbe Ursache betroffen —
  die Kreis-Rasterisierung war schon zellzentriert und flächentreu genug (Fehler
  eine Größenordnung kleiner als bei den Rechtecken, per Verfeinerung
  bestätigt). Dort ist Moose schlicht noch nicht konvergiert (monotoner Anstieg
  mit der Ordnung, sauberer 1/m-Fit), kein Rasterisierungsproblem.
- Nebenbefund: die `(0,0)`-Punkte (ein einziger Retained Order) in
  `moose_reference.json` sind unphysikalisch/kaputt — bei einer Ordnung muss
  RCWA exakt auf das flächengemittelte effektive Medium reduzieren, tut es bei
  diesen gespeicherten Werten aber nachweislich nicht (Moose-Breitensweep bei
  m=0 zeigt: das Ergebnis reagiert dort gar nicht auf Änderungen der
  Atom-Geometrie). Diese Punkte sollten nicht weiterverwendet werden.

**3. Was war fehlerhaft / nicht weiter verfolgen.**

- Erste Vermutung, Moose würde subpixel-genau/flächengewichtet rastern
  (Begründung: Moose lag näher am "korrekten" Wert als die Python-Maske) —
  widerlegt durch `moose_geometry_probe.cs` Probe A: Moose rastert ebenfalls
  strikt binär (immer genau 2 Permittivitätswerte im gerenderten Grid), nur mit
  anderer Rundungsrichtung/anderem Offset als `structures.py`. Nicht nochmal als
  Erklärung annehmen.
- Nicht weiter der Spur nachgehen, dass unterschiedliche Faktorisierungsregeln
  (Laurent/Pol/Li/Normal-Vector) die Ursache sein könnten — das wurde geprüft
  und ausgeschlossen, da mehrere unabhängige Regeln in Python untereinander
  übereinstimmen, aber gemeinsam von Moose abweichen; die Regel-Wahl ist also
  nicht der Hebel.
- Der ursprüngliche `FFT_MODE=1`-Verfeinerungsmechanismus im Skript
  (`moose_convergence_bench.cs`) mit den alten Grenzwerten ist erwiesenermaßen
  keine funktionierende Nachbildung von grcwas festem 256er-Gitter auf der
  Moose-Seite — nicht ungeprüft davon ausgehen, dass dieser Modus tut, was der
  Name/Kommentar suggeriert; die tatsächlichen Grenzen von Moose sind
  maßgeblich.

**4. Wo wir stehen geblieben sind.**

Unmittelbar nächster Schritt, der direkt angeschlossen hätte: der
Nutzer hatte begonnen, mit dem korrigierten `moose_geometry_probe.cs`
zusätzliche Vergleichspunkte (weitere Ordnungen/Verfeinerungsstufen, u.a. für
`C1b`/`C2`) manuell in der Moose-UI nachzurechnen, um die Übereinstimmung mit
Python noch breiter abzusichern; das war nicht abgeschlossen, als der Chat
endete.

Übergeordnete, noch nicht begonnene Richtung: die eigentliche Korrektur der
Rasterisierung in `structures.py` (zellzentrierte Abtastung / exakt passendes
Gitter für den Rechteck-Zweig, analog zum bereits korrekten Kreis-Zweig) wurde
in diesem Chat bewusst NICHT vorgenommen, weil sie sämtliche bestehenden
2D-Referenzwerte in `moose_reference.json`/`ikarus_reference.json` invalidiert
und einen kompletten Neulauf beider Suiten erfordert. Das ist der vorgeschlagene
nächste große Schritt, sobald das Zusammenführen der Branches abgeschlossen
ist. Ebenso offen: die kaputten `(0,0)`-Referenzpunkte aus
`moose_reference.json` entfernen/korrigieren, und die Moose-Skript-Konstanten
(Verfeinerungsgrenzen) endgültig gegen den tatsächlichen Moose-Wertebereich
festschreiben statt gegen Annahmen.
