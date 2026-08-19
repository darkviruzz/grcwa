# Handoff notes

Compact per-branch summaries for a later merge session. Each section is
append-only — do not edit or remove another branch's section.

## claude/rcwa-game-new-player-a3lpbu

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
