# Handoff notes

Compact per-branch summaries for a later merge session. Each section is
append-only — do not edit or remove another branch's section.

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
