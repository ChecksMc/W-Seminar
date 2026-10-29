#set document(
  title: "CNN-gestützte Analyse emotionaler Reaktionen auf Kunstwerke",
  author: "W-Seminar",
)
#set page(margin: 2.5cm)
#set text(lang: "de", font: "Libertinus Serif", size: 11pt)
#set par(justify: true, leading: 0.72em)

= Methodik: Emotionserkennung und KNN-basierter Merkmalsraum

== Ziel und Abgrenzung

In diesem Projekt wird ein Convolutional Neural Network (CNN) trainiert, das
Kunstwerke anhand ihrer Bilddaten einer der dominanten Emotionsklassen von
EmoArt zuordnet. Die Vorhersage beschreibt die im Datensatz annotierte Emotion.
Sie ist keine direkte Messung des Erlebens einer bestimmten Person. EmoArt
umfasst multidimensionale Beschreibungen von Kunstwerken und emotionale Annotationen
[@zhang2025emoartmultidimensionaldatasetemotionaware].

Ein zusätzliches Ziel ist eine interaktive dreidimensionale Darstellung, in
der ähnliche Werke anhand ihrer visuellen Merkmale und ihrer vorhergesagten
Emotionen benachbart liegen. Die drei Achsen werden nicht vorab mit Valenz,
Erregung oder einem Damasio-Begriff gleichgesetzt. Ihre Positionen entstehen
aus den Beziehungen zwischen den vom CNN erkannten Werken.

== Datengrundlage und Aufteilung

Verwendet wird die kompakte
#link("https://huggingface.co/datasets/printblue/EmoArt-5k")[EmoArt-5k-Version
auf Hugging Face]. Die lokal heruntergeladene Fassung enthält 5.532 Annotationen
und 5.532 zugehörige Bilddateien. Die Annotationen enthalten unter anderem eine dominante Emotion,
eine positive oder negative Valenz und ein niedriges oder hohes
Erregungsniveau. Für die hier beschriebene Klassifikation dient die dominante
Emotion als Lernziel. Valenz und Erregung werden nicht als feste
Raumkoordinaten verwendet.

Die Bilder werden als RGB-Dateien eingelesen, auf 128 × 128 Pixel skaliert und
kanalweise auf den Wertebereich $[-1, 1]$ normiert. Die Daten werden mit einem
festen Zufallsstartwert in Trainings-, Validierungs- und Testdaten aufgeteilt.
Das angestrebte Verhältnis beträgt 70:15:15. Bei 5.532 Bildern entspricht dies
ungefähr 3.872 Trainingsbildern sowie jeweils 830 Validierungs- und Testbildern.
Die Aufteilung wird als CSV-Datei gespeichert, damit weitere Versuche dieselben
Bilder in denselben Teilmengen verwenden.

Die Aufteilung wird nach den gemeinsamen Valenz-Erregungs-Kombinationen
stratifiziert. Seltene Emotionsklassen werden, falls nötig, mit mindestens einem
Beispiel in der Trainingsmenge gehalten. Die Klassen sind ungleich häufig
vertreten. Deshalb verwendet das Training inverse Häufigkeitsgewichte in der
Verlustfunktion und bewertet die Ergebnisse zusätzlich mit Makro-F1 und
balancierter Genauigkeit.

== Architektur und Training

Das CNN verarbeitet drei Farbkanäle. Vier Faltungsschichten vergrößern die
Zahl der Merkmalskanäle schrittweise von 16 über 32 und 64 auf 96. Auf die
Faltungen folgen ReLU-Aktivierungen; die ersten drei Blöcke enthalten jeweils
eine Max-Pooling-Operation. Adaptives Average-Pooling fasst die räumlichen
Merkmale zusammen. Eine dichte Schicht erzeugt daraus einen 64-dimensionalen
Bildvektor. Der Klassifikationskopf gibt für jede der zwölf Emotionsklassen
einen Logit aus.

Für die Logits $s_k$ ergibt Softmax die Klassenwahrscheinlichkeit

$ p_k = exp(s_k) / sum_(j=1)^C exp(s_j), quad k in {1, ..., C}. $

Das CNN wird mit zufälliger Initialisierung trainiert; es werden keine externen
vortrainierten Bildgewichte verwendet. Optimiert wird eine gewichtete Kreuzentropie:

$ L = -sum_(i=1)^N w_(y_i) log(p_(i,y_i)), $

wobei $y_i$ die annotierte Klasse und $w_(y_i)$ ihr aus der Trainingsmenge
berechnetes Häufigkeitsgewicht bezeichnet. Die Gewichte werden nicht aus dem
Testdatensatz abgeleitet. Als Optimierer wird AdamW eingesetzt.

Die Hyperparametersuche vergleicht vier Kombinationen aus zwei Lernraten
($10^(-3)$ und $3 dot 10^(-4)$) und zwei Dropout-Raten (0,2 und 0,4). Die
Validierungsmenge wählt die beste Konfiguration anhand des Makro-F1-Werts aus.
Die Testmenge bleibt bis zur abschließenden Auswertung unberührt. Berichtet
werden Genauigkeit, balancierte Genauigkeit, Makro-F1 und der Testverlust.

== Konstruktion des dreidimensionalen KNN-Raums

Für jedes bekannte Bild liefert das trainierte CNN zwei Repräsentationen: den
64-dimensionalen Bildvektor $z_i$ und die Verteilung $p_i$ über die
Emotionsklassen. Beide Vektoren werden jeweils auf Einheitslänge normiert und
mit gleichem Gewicht zu einem gemeinsamen Merkmalsvektor verbunden:

$ tilde(z)_i = z_i / norm(z_i), quad tilde(p)_i = p_i / norm(p_i), $

$ u_i = norm(0.5 tilde(z)_i oplus 0.5 tilde(p)_i). $

Die Ähnlichkeit zweier Werke wird über den Kosinusabstand ihrer kombinierten
Merkmalsvektoren bestimmt:

$ d_(ij) = 1 - u_i dot u_j. $

Für jedes Werk werden die zwölf nächsten Nachbarn gesucht. Aus diesen
Nachbarschaften entsteht ein symmetrischer gewichteter Graph. Die Kanten werden
mit $A_(ij) = exp(-(d_(ij) / s)^2)$ gewichtet; $s$ ist der Median der positiven
Nachbarabstände. Ein spektrales Embedding projiziert den Graphen in drei
Dimensionen; spektrales Clustering ordnet die Werke zusätzlich Gruppen zu. In
der interaktiven Plotly-Grafik steht jede Markierung für ein Kunstwerk. Die
Farbe zeigt die KNN-Gruppe; beim Überfahren mit dem Mauszeiger erscheinen unter
anderem die vorhergesagte und die annotierte Emotion, der Kunststil und die
Vorhersagesicherheit.

Ein neues Bild erhält zunächst eine Emotionsverteilung und einen Bildvektor.
Anschließend wird es mit den bekannten Werken verglichen. Seine dargestellte
Position wird als invers abstandsgewichteter Mittelwert der Koordinaten seiner
nächsten Nachbarn berechnet. Die Gruppenzuordnung entspricht der häufigsten
Gruppe unter diesen Nachbarn.

Die resultierenden Koordinaten sind eine datengetriebene Visualisierung. Die
Richtung einer Achse besitzt keine festgelegte psychologische oder biologische
Bedeutung. Auch die KNN-Gruppen sind keine eigenständigen psychologischen
Kategorien; sie fassen Bilder mit ähnlichen CNN-Repräsentationen zusammen.

== Reproduzierbarkeit und Ausführung

Der Quellcode unterscheidet automatische Gerätewahl, CPU-Betrieb und
NVIDIA-CUDA-Betrieb. Ohne Argument wird eine verfügbare CUDA-fähige NVIDIA-GPU
gewählt; andernfalls wird die CPU verwendet. Eine nicht verfügbare, ausdrücklich
angeforderte CUDA-Ausführung führt zu einer Fehlermeldung.

```sh
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r cnn-requirements.txt
python -m src.CNN.download_data
python -m src.CNN.train --device cpu
python -m src.CNN.train --tune --epochs 8 --device cpu
python -m src.CNN.predict path/to/artwork.jpg --device cpu
python -m src.CNN.visualize --device cpu
```

Auf einem System mit funktionsfähiger NVIDIA-CUDA-Installation kann für
Training, Vorhersage und Visualisierung jeweils `--device nvidia` oder
`--device cuda` angegeben werden. `--device auto` wählt die verfügbare
Ausführungseinheit automatisch. Der CPU-Befehl zur Installation von PyTorch
installiert keine CUDA-Unterstützung. Für NVIDIA-Betrieb muss deshalb eine
zur Hardware passende CUDA-Version von PyTorch installiert werden. Die passende
Installationsvariante kann über die
#link("https://pytorch.org/get-started/locally/")[offizielle Installationsauswahl
von PyTorch] ausgewählt werden.
Die Modellgewichte, Trainingsberichte und interaktive Grafik werden lokal im
Verzeichnis `runs/` abgelegt.

== Einschränkungen

Die Emotionen stammen aus den EmoArt-Annotationen. Sie geben nicht die
individuelle Reaktion einer Versuchsperson wieder. Dafür wäre eine eigene
Erhebung mit mehreren Betrachterinnen und Betrachtern erforderlich. Außerdem
kann eine zufällige Bildaufteilung dazu führen, dass Werke desselben Stils oder
desselben Künstlers sowohl in Trainings- als auch in Testdaten liegen. Die
ausgewiesenen Testwerte belegen daher nicht automatisch eine Übertragbarkeit
auf unbekannte Kunststile, kulturelle Kontexte oder Personen.

Die Bilddateien und Annotationen unterliegen eigenen Nutzungsbedingungen. Die
MIT-Lizenz dieses Projekts bezieht sich ausschließlich auf den hier
geschriebenen Quellcode. Vor einer Weitergabe der Datendateien müssen die
geltenden Lizenzen und Quellen der einzelnen Kunstwerke geprüft werden.

#bibliography("sources.bib", style: "apa")
