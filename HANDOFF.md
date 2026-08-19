# Handoff notes

Per-branch summaries for a later merge session. Each section is a self-contained
account of one branch's work — no cross-referencing assumed.

## claude/structure-convergence-viz-407ocl

**Ziel:** Visuelle Darstellung für die 13-Struktur-Benchmark-Batterie und für die
Konvergenzstudie (`night_run_2`) entwickeln. Ausgangslage: keine Strukturbilder
existierten; die Konvergenzplots (`plot_conv.py`, `plot_moose.py`) waren schwer
lesbar (rot/grün-Kodierung, oszillierende Rohkurven, Trennung in raw/tight-Paare).
Vorgehen war explizit iterativ mit Nutzer-Feedback über mehrere Mockup-Runden
statt eine fertige Lösung zu entwerfen.

**Was funktioniert hat:**
- Mehrere Stilvarianten parallel als PNG-Mockups vorgelegt (nicht nur eine
  Lösung), dann per Nutzer-Feedback verfeinert. Das war der eigentliche
  Arbeitsmodus dieses Chats.
- Für Strukturen: isometrische Unit-Cell-Darstellung mit echten 3D-Festkörpern
  (nicht texturierte Blöcke) — Luft wird als *Abwesenheit* von Material gezeichnet
  (leerer Raum zwischen Stegen/Pillars), nicht als grau gefüllte Fläche. Freistehende
  Fälle (Substrat = Luft) bekommen ein gestricheltes Phantom-Wireframe statt eines
  soliden Blocks; dessen Deckfläche muss mitgezeichnet werden, sonst wirkt die
  Struktur wie freischwebend. 1D- und 2D-Strukturen bekommen dieselbe Zellzahl
  (nper×nper), damit die Tiefe (y) über das ganze Blatt konsistent ist.
- Für Konvergenz: Farbe = Faktorisierungsregel (Laurent/Pol/Li/NV), Strich+Marker
  = Codebasis (fork/ikarus) — invertiert bewusst die bisherige Konvention
  (Farbe=Codebasis). Okabe-Ito-Palette statt rot/grün. Zwei Darstellungen wurden
  vom Nutzer als klar überlegen ausgewählt: (a) "cost staircase" — laufendes
  Minimum des Fehlers über Wall-Time, ersetzt die verrauschte
  `conv_accuracy_vs_time.png`; (b) signierte Abweichung auf Symlog-Achse
  (linear innerhalb ±1e-4, sonst log) — ersetzt das `moose_compare_raw` /
  `_tight`-Paar in einer Figur und erhält das Vorzeichen der Annäherung.
- Gesamte Bildpipeline ist reproduzierbar aus dem Repo heraus lauffähig gemacht
  (nicht nur einmalig von Hand zusammengesetzt): `plot_structures.py`,
  `plot_conv_cost.py`, `plot_deviation.py` als reguläre Plotter in `benchmark/`,
  plus ein HTML-"Plate Book" als interaktives Review-Tool
  (`viz_proposals/build_plate_book.py`, Templates in `viz_proposals/plate_book/`),
  aus Templates + gerenderten Figuren + JSON-Payload zusammengesetzt, keine
  Handarbeit mehr nötig.
- Cherry-Pick eines parallelen Branches (Moose-Neurechnung) hat aufgedeckt: die
  Sweep-Key-Konvention in `moose_reference.json` änderte sich (Key ist Mooses
  Maximalordnung m, nicht Ordnungsanzahl — 1D: 2m+1, 2D: `(mx,my)` →
  (2mx+1)(2my+1)). Der eigene Parser (`viz_conv._mkey`) war falsch und wurde
  gegen `moose_timing.json`s mitgeliefertes `nG` pro Punkt verifiziert (0
  Abweichungen). `plot_moose.py` hatte diese Konvention schon korrekt.

**Was fehlerhaft war / nicht weiterverfolgen:**
- Erste Version der isometrischen Struktur-Darstellung zeichnete Luft als
  hellgraue Fläche (texturierter Block statt echter Festkörper) — vom Nutzer
  explizit verworfen zugunsten von "Luft = leerer Raum".
- Zylinder-Rendering (Kreis-Pillar D2) hatte zuerst ein sich selbst
  überschneidendes Polygon für die Mantelfläche — sichtbar als kaputte
  C-Form statt Zylinder. Gefixt durch Streifen-Rendering mit
  Beleuchtungsgradient statt einem einzelnen Polygonzug.
- S1 (bemaßter x-z-Schnitt, "mechanical drawing"-Stil) wurde bewusst NICHT
  in die Plotter-Pipeline übernommen — Maßangaben sind bei Blattgröße schlecht
  lesbar. Bleibt als Proposal in `viz_proposals/`, nicht weiterentwickeln ohne
  vorher das Bemaßungs-Layout zu überarbeiten (vermutlich Tabelle statt
  Pfeile im Bild).
- Digit-Map (C1, Konvergenz als "korrekte Nachkommastellen"-Heatmap) und
  Scoreboard (C3, Kosten-bis-1e-4 pro Fall×Spalte) wurden vom Nutzer nicht für
  die Haupt-Pipeline ausgewählt — bleiben als Proposals, nicht adoptiert.
- Ursprüngliche Annahme "Moose trägt keine Zeiten, kann nicht auf der
  Zeitachse erscheinen" war nur für den alten Datenstand richtig — nach dem
  Cherry-Pick hat Moose `t_solve` pro Punkt und gehört auf beide Achsen.
- Referenzwerte in `conv_results.json` sind zum Laufzeitpunkt eingebacken;
  `moose_reference.json` hat sich seitdem unabhängig weiterbewegt. Für
  B2/B3/C1/C1b weichen die beiden bis zu 8e-4 voneinander ab (über der
  1e-4-Tolerenz der Studie). Die neuen Konvergenz-Plotter lesen den
  *aktuellen* Moose-Wert nach; `plot_conv.py`/`plot_moose.py` nutzen weiterhin
  den eingebackenen Wert. Beide Figurensätze widersprechen sich also aktuell
  bei diesen vier Fällen — nicht als Bug in diesem Branch behandeln, sondern
  als offene Frage, ob `conv_run.py` neu laufen soll.

**Wo wir stehen geblieben sind:**
Unmittelbar nächster Schritt (unterbrochen durch diesen Handoff-Auftrag): keiner
angestoßen — die letzte inhaltliche Aktion war der Push von Commit `1261bdd`
(Moose-Konventions-Fix + Referenz-Refresh + Zeiten auf beiden Achsen) und die
Aktualisierung des Plate-Book-Artifacts; der Nutzer hat dann nur das Modell
gewechselt.

Übergeordnete, vom Nutzer noch nicht in Auftrag gegebene, aber im Gespräch
skizzierte Richtung:
- S1 (bemaßter Schnitt) überarbeiten, falls doch gebraucht.
- Entscheiden, ob die vier stale-reference-Fälle einen Re-Run von
  `conv_run.py` auslösen, oder ob `plot_conv.py`/`plot_moose.py` ebenfalls auf
  den live gelesenen Moose-Wert umgestellt werden sollen, um die beiden
  Figurensätze wieder konsistent zu machen.
- Die geometrische Rasterisierung selbst (FFT-Grid, Kantenpixel, effektives
  Medium, erzwungene Gittergrenzen) war in diesem Chat kein Thema — das
  vom Nutzer gepushte Sibling-Branch-Ergebnis (Moose-Neuvermessung) wurde nur
  als Datenlieferant cherry-gepickt, seine Methodik nicht geprüft. Die
  `_provenance`-Notiz in der gemergten `moose_reference.json` weist selbst
  darauf hin, dass die 2D-Fälle (C1, C1b, C2, D2) bei den aktuell erreichten
  Ordnungen noch nicht konvergiert sind und teils bis zu ~1e-2 von den
  faithful-rule-Werten (Pol/Li/NV) abweichen — das ist im Datenstand sichtbar,
  aber in diesem Chat nicht inhaltlich untersucht worden.
