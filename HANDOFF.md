# Handoff notes

Compact per-branch summaries from the four benchmark/convergence sessions,
collected here for the merge session. Sections are unedited, concatenated
in the chronological order the branches were worked on (oldest first).

## claude/rcwa-game-new-player-a3lpbu (claude-opus-5)

**Ziel:** Der User hatte von einem neuen RCWA-Solver "Ikarus" (CAVITY
technologies GmbH, Whitepaper mit DOI) erfahren, dessen Whitepaper grcwa
explizit als Beispiel für einen fehlerhaft konvergierenden "direct rule"
(Laurent) Solver in TM-Polarisation nennt. Aufgabe: diesen Kandidaten
selbstständig in den bestehenden `benchmark/`-Vergleichsrahmen einbauen,
die Behauptungen des Whitepapers nachrechnen/verifizieren, und eine
Bewertung abgeben.

**Was hat funktioniert:**
- Ikarus ist als `ikarus-rcwa` frei auf PyPI installierbar und wurde live
  eingebunden (nicht nur als statische Referenzzahlen), über einen neuen
  Adapter `benchmark/ikarus_suite.py`, der Ikarus' komplett anderes
  Konstrukt (SI-Einheiten, halbunendliche Cover/Substrat-Layer,
  Integer-Topologie+Materialliste, `n_orders` als Maximalordnung pro Achse,
  `simulate()` gibt T zuerst zurück) auf dieselben Battery-Strukturen
  abbildet wie grcwa.
- Zentraler Fix für Vergleichbarkeit: die geometrische Rasterisierung jeder
  strukturierten Schicht wurde in `structures.py` in eine einzige Funktion
  `layer_mask()` gezogen, die von *beiden* Backends (grcwa und Ikarus)
  konsumiert wird — vorher hätte jedes Backend sein eigenes Pixelgrid
  gebaut, was eine Cross-Code-Abweichung fälschlich als "Geometrie-
  Artefakt" statt als echten Faktorisierungs-Unterschied hätte erscheinen
  lassen. Das war explizit notwendig, weil sonst nicht unterscheidbar
  gewesen wäre, ob eine Diskrepanz an der FFT-Rasterisierung/Kantenpixeln
  liegt oder an der Fourier-Faktorisierungsregel selbst.
- Whitepaper-Behauptungen wurden alle reproduziert und bestätigt (Direct-
  Rule-Fehler in TM, O(1/M)-Konvergenz, Energieerhaltung ≠ Konvergenztest,
  normal-vector-Methode schlägt Li's separable rule auf gekrümmten
  Rändern) — alle mit eigenem Skript `benchmark/ikarus_whitepaper_check.py`
  nachvollziehbar, 8/8 Checks grün.
- Wichtiger Gegenbefund: die im Whitepaper genannte grcwa-Zahl (17.5%
  Fehler) gilt nur für das unfixte Upstream-Laurent; der bereits in diesem
  Fork gefixte Pol-Modus konvergiert korrekt auf den faithful-Wert (0.1001
  vs. 0.100 published) und verhält sich auf gekrümmten Rändern wie eine
  normal-vector-Methode, nicht wie Li's separable rule.
- Performance-Vergleich sauber getrennt: pro-Solve-Zeit bei fixem nG
  (Ikarus 2-5x langsamer) vs. Zeit-bis-Zielgenauigkeit (Ikarus dort ~35x
  schneller, weil es viel weniger Ordnungen braucht) — beides gemessen,
  nicht nur behauptet.

**Was war fehlerhaft / nicht weiter verfolgen:**
- Direkter Download der Whitepaper-PDF per WebFetch/curl von
  `cavity-technologies.com` schlug fehl — der Host ist über die
  Egress-Proxy-Policy dieser Session blockiert (403 auf CONNECT). Nicht an
  Netzwerk-Workarounds für diesen Host weiterarbeiten; die PDF kam
  stattdessen vom User als Upload.
- PDF-Text-Extraktion über `pypdf` scheiterte zunächst an einer kaputten
  `cffi`/`cryptography`-Systeminstallation (pyo3-Panic beim Import). Nicht
  erneut mit System-`pypdf` versuchen — Workaround war ein frisches venv
  mit `pdfminer.six`, das zuverlässig lief.
- Keine Sackgassen bei der eigentlichen Solver-Integration oder den
  physikalischen Ergebnissen — alle Ansätze dort haben sich bestätigt und
  wurden nicht verworfen.

**Wo wir stehen geblieben sind:**
Die Arbeit ist inhaltlich abgeschlossen und committet/gepusht (5 Commits
auf diesem Branch, kein PR angelegt). Unmittelbar nächster Schritt wäre
gewesen, auf expliziten Wunsch des Users einen PR zu öffnen — das wurde
nicht angefordert und ist offen. Übergeordnet als sinnvolle Fortsetzung
identifiziert, aber noch nicht begonnen: der Fork-eigene Pol-Modus
konvergiert zwar korrekt, oszilliert aber bis zu deutlich höheren
Ordnungszahlen als Ikarus' normal-vector-Methode, bevor er sich setzt
(Faktor ~10 in Ordnungszahl, ~2 Größenordnungen in Zeit auf dem
Referenzfall D1) — das wurde als konkrete, noch offene Verbesserungs-
Baustelle für den Fork benannt, aber nicht angegangen.


## claude/rcwa-game-new-player-a3lpbu (GPT-5.6-Sol)

**Ziel:** Den Abbruch des gestuften Konvergenzlaufs beheben: Ikarus
erkannte bei den abgeleiteten hohen eindimensionalen Harmoniken ein zu
grobes gemeinsames FFT-Raster. Der zugehörige Fix ist `fc12bcd`
(`fix(benchmark): enforce FFT grid resolution for convergence sweeps`).

**Was hat funktioniert:**
Die Ursache war die fehlende Abtastung aller Fourier-Differenzordnungen,
nicht ein beschädigter Cache. Das gemeinsame eindimensionale Geometrieraster
wurde für den vollständigen Lauf ausreichend vergrößert und eine frühe
Prüfung für beide aktiven Achsen vor Worker- und Solverstart ergänzt. Das
Festhalten derselben Maske und Rasterisierung für grcwa und Ikarus bleibt
die richtige Grundlage für einen fairen Solververgleich. Nach der
physikalisch relevanten Rasteränderung sorgen neue Cache-Fingerprints
korrekt für eine Neuberechnung.

**Was war fehlerhaft / nicht weiter verfolgen:**
Ein unveränderter Neustart hätte denselben Fehler wiederholt. Die Ikarus-
Prüfung darf nicht umgangen und nur Ikarus darf nicht dynamisch
hochskaliert werden; sonst vergleichen die Solver unterschiedliche Raster,
während grcwa hohe Ordnungen unter Umständen still aliasiert. Ebenso dürfen
alte eindimensionale Physikpunkte nach der Rasteränderung nicht in die neue
Konvergenzkurve übernommen werden. Spätere Energieerhaltungswarnungen der
Ikarus-Normalvektormethode bei schwierigen zweidimensionalen Strukturen
waren niedrigordentliche numerische Instabilität, nicht derselbe
Rasterfehler und kein Prozessabbruch.

**Wo wir stehen geblieben sind:** Der Fix war getestet und der Nachtlauf
mit neuen Cache-Fingerprints neu gestartet; zuletzt lief er weiter,
während die Ikarus-Energiewarnungen eingeordnet wurden. Unmittelbar als
Nächstes wäre zu bestätigen gewesen, dass der Lauf vollständig beendet ist,
und anschließend den neu erzeugten Ergebnissatz zu prüfen. Grundsätzlich
sollte das gemeinsame Raster samt Vorabprüfung als Invariante beibehalten
werden; getrennt davon kann später entschieden werden, ob endliche, aber
energieverletzende Niedrigordnungspunkte als sichtbare Konvergenzausreißer
bleiben oder einen eigenen Qualitätsstatus erhalten.



## claude/rcwa-game-new-player-a3lpbu (GPT-5.6-Sol)

**Ziel:** Den RCWA-Konvergenzbenchmark auf Konvergenzgeschwindigkeit,
belastbare Laufzeitabschätzung und den Vergleich von fork, Ikarus und Moose
ausrichten. Dazu sollten Wiederholungsaufwand reduziert, lange Läufe
fortsetzbar, wachsende Zwischenstände plotbar und Referenzen sinnvoll
priorisiert werden. Der zentrale Ergebnis-Commit dieses Chats ist `dccf7ae`.

**Was hat funktioniert:** Ein atomarer Punkt-Cache mit getrennten
Identitäten für Physik und maschinenabhängige Zeiten erlaubt Fortsetzen
und Erweitern ohne erneute bekannte Lösungen. Der ergebnisliefernde Solve
wird mitgemessen; nur schnelle Punkte werden wiederholt und über das
Minimum bewertet. Rohzeiten bleiben erhalten, während gruppierte monotone
Kurven Laufzeiten schätzen. Kumulative dichte Ordnungsgitter aktualisieren
die Konvergenzplots stufenweise; für eindimensionale Fälle bewahrt die
Vereinigung aus direkter Ordnung und deren Quadrat sowohl dichte kleine
als auch vergleichbare hohe Gesamtordnungen. Moose ist bevorzugte Referenz,
sonst Ikarus NV; unvollständige Sweeps werden vor Konvergenzaussagen
abgewiesen. Enge Rohwertplots funktionieren mit natürlichem Clipping,
ohne außerhalb liegende Punkte zu entfernen.

**Was war fehlerhaft / nicht weiter verfolgen:** Der separate Single-
Order-Lauf war redundant. Ein ungemessener erster Ergebnis-Solve plus
zusätzliche Timing-Läufe verschwendete Arbeit; ebenso war der Median als
Wert für Wiederholungen nicht gewünscht. Punkte außerhalb enger Plotfenster
auszublenden erzeugte irreführende Linien, und ein relatives Prozentfenster
war die falsche Interpretation; richtig ist ein absoluter Rohwertausschnitt
ohne Datenfilterung. Ein Quick-Zwischenstand mit nur einer Ordnung ließ das
Plotraster ohne Zeilen scheitern. Die eindimensionale Liste lediglich an
die per-Achse-Liste zu spiegeln kappte ihre Gesamtordnung zu früh; die
abgeleitete Vereinigung mit den Quadraten behebt das. Gepufferte Worker-
Ausgabe ließ lange Läufe scheinbar hängen, während still herausgefilterte
Fehler unvollständige Sweeps fälschlich erfolgreich erscheinen ließen. Auch
ein fehlgeschlagenes Timing-Refresh darf bereits gecachte Physik nicht
verwerfen.

**Wo wir stehen geblieben sind:** Der Umbau war implementiert, mit
fokussierten Tests und kalten sowie vollständig gecachten Quick-Durchläufen
geprüft und in `dccf7ae` festgehalten; der eigentliche lange Stufenlauf
wurde bewusst nicht gestartet. Unmittelbar als Nächstes wäre dieser
Langlauf zu beobachten und anschließend die fortgeschriebenen Konvergenz-
und Cache-Artefakte auszuwerten. Die übergeordnete Richtung bleibt:
schwierige Strukturen nach erreichbarer Genauigkeit und geschätzter Zeit
bis zur Konvergenz bewerten, Moose als unabhängige Referenz nutzen und
andernfalls Ikarus NV heranziehen.


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
