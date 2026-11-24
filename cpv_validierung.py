
import pandas as pd
import re

# Offizielles CPV-Verzeichnis öffnen
excel_datei = pd.ExcelFile("cpv_2008_ver_2013.xlsx")

# Verfügbare Tabellenblätter anzeigen
print("Tabellenblätter:")
print(excel_datei.sheet_names)

# Erstes Tabellenblatt einlesen
cpv_df = pd.read_excel(excel_datei, sheet_name=0)

print("\nAnzahl der Datensätze:")
print(len(cpv_df))

print("\nSpaltennamen:")
print(cpv_df.columns.tolist())

print("\nErste fünf Datensätze:")
print(cpv_df.head())


# Relevante Spalten auswählen
cpv_df = cpv_df[["CODE", "DE"]].copy()

# Achtstelligen CPV-Hauptcode extrahieren
cpv_df["cpv_code"] = cpv_df["CODE"].str.extract(
    r"^(\d{8})-\d$"
)

# Deutsche Beschreibung umbenennen
cpv_df = cpv_df.rename(columns={"DE": "beschreibung"})

print("\nAufbereitetes CPV-Verzeichnis:")
print(cpv_df.head(10).to_string(index=False))

print("\nFehlende CPV-Hauptcodes:")
print(cpv_df["cpv_code"].isnull().sum())

print("\nEindeutige CPV-Hauptcodes:")
print(cpv_df["cpv_code"].nunique())

# Gültige CPV-Hauptcodes als Set speichern
gueltige_cpv_codes = set(cpv_df["cpv_code"].dropna())

print("\nAnzahl gültiger CPV-Hauptcodes:")
print(len(gueltige_cpv_codes))

# Ausschreibungsdatensatz einlesen
df = pd.read_csv("ausschreibungen_2023_1000.csv")

# Funktion zur CPV-Extraktion und Validierung
def extrahiere_cpv_codes(text):
    kandidaten = re.findall(r"\b\d{8}(?:-\d)?\b", text)

    validierte_codes = set()

    for kandidat in kandidaten:
        hauptcode = kandidat[:8]

        if hauptcode in gueltige_cpv_codes:
            validierte_codes.add(hauptcode)

    return sorted(validierte_codes)


# Funktion auf alle Ausschreibungstexte anwenden
df["cpv_codes"] = df["text"].apply(extrahiere_cpv_codes)

# Anzahl validierter Codes je Ausschreibung
df["cpv_anzahl"] = df["cpv_codes"].apply(len)

# Ergebnisse ausgeben
print("\nErgebnisse der CPV-Validierung:")
print("Ausschreibungen insgesamt:", len(df))
print("Mit gültigen CPV-Hauptcodes:", (df["cpv_anzahl"] > 0).sum())
print("Ohne gültige CPV-Hauptcodes:", (df["cpv_anzahl"] == 0).sum())

print("\nAnzahl unterschiedlicher gefundener CPV-Codes:")
print(len(set(df["cpv_codes"].explode().dropna())))

print("\nErste zehn Ausschreibungen:")
print(df[["filename", "cpv_codes", "cpv_anzahl"]].head(10).to_string(index=False))


# Relevante CPV-Bereiche für das IT-Systemhaus
it_bereiche = ("30", "32", "48", "50", "72")

# IT-Klassifikation anhand validierter CPV-Codes
def ist_it_relevant(codes):
    return any(code.startswith(it_bereiche) for code in codes)


df["it_kandidat_validiert"] = df["cpv_codes"].apply(
    ist_it_relevant
)

# Vergleich der bisherigen und neuen Klassifikation
print("\nVergleich der IT-Klassifikation:")

print("IT-Kandidaten vorher:", df["it_kandidat"].sum())
print("IT-Kandidaten nachher:", df["it_kandidat_validiert"].sum())

print("\nVerteilung nach der Validierung:")
print(df["it_kandidat_validiert"].value_counts())

# Durch die Validierung ausgeschlossene IT-Kandidaten
ausgeschlossen = df[
    df["it_kandidat"] & ~df["it_kandidat_validiert"]
]

print("\nAusgeschlossene IT-Kandidaten:", len(ausgeschlossen))

for _, zeile in ausgeschlossen.iterrows():
    print("\n----------------------------")
    print("Dateiname:", zeile["filename"])
    print("Gültige CPV-Codes:", zeile["cpv_codes"])
    print("Text:", zeile["text"][:300])


# IT-Kandidaten auswählen
it_df = df[df["it_kandidat_validiert"]].copy()

# IT-relevante CPV-Codes extrahieren
it_df["it_cpv_codes"] = it_df["cpv_codes"].apply(
    lambda codes: [
        code for code in codes
        if code.startswith(it_bereiche)
    ]
)

# Häufigkeit der einzelnen IT-CPV-Codes ermitteln
cpv_haeufigkeit = (
    it_df["it_cpv_codes"]
    .explode()
    .value_counts()
    .rename_axis("cpv_code")
    .reset_index(name="anzahl")
)

# Offizielle deutsche Beschreibungen hinzufügen
cpv_haeufigkeit = cpv_haeufigkeit.merge(
    cpv_df[["cpv_code", "beschreibung"]],
    on="cpv_code",
    how="left"
)

print("\nGefundene IT-CPV-Codes mit Beschreibung:")
print(cpv_haeufigkeit.to_string(index=False))


# Präzisierte CPV-Bereiche für das IT-Systemhaus
it_bereiche_praezise = ("302", "324", "48", "503", "72")

# Präzisierte IT-Klassifikation
def ist_it_relevant_praezise(codes):
    return any(
        code.startswith(it_bereiche_praezise)
        for code in codes
    )


df["it_kandidat_praezise"] = df["cpv_codes"].apply(
    ist_it_relevant_praezise
)

# Vergleich aller drei Klassifikationsstufen
print("\nVergleich aller Klassifikationsstufen:")
print("Ursprünglich:", df["it_kandidat"].sum())
print("Nach CPV-Validierung:", df["it_kandidat_validiert"].sum())
print("Nach Präzisierung:", df["it_kandidat_praezise"].sum())

# Ausgeschlossene Kandidaten untersuchen
ausgeschlossen_praezise = df[
    df["it_kandidat_validiert"] &
    ~df["it_kandidat_praezise"]
]

print("\nZusätzlich ausgeschlossene Kandidaten:")
print(len(ausgeschlossen_praezise))

for _, zeile in ausgeschlossen_praezise.iterrows():
    print("\n----------------------------")
    print("Dateiname:", zeile["filename"])
    print("CPV-Codes:", zeile["cpv_codes"])
    print("Text:", zeile["text"][:300])


# Präzisierte IT-Kandidaten auswählen
it_praezise_df = df[df["it_kandidat_praezise"]].copy()

# Zufällige, reproduzierbare Stichprobe
stichprobe = it_praezise_df.sample(
    n=10,
    random_state=42
)

print("\nStichprobe der verbleibenden IT-Kandidaten:")

for _, zeile in stichprobe.iterrows():
    print("\n----------------------------")
    print("Dateiname:", zeile["filename"])
    print("CPV-Codes:", zeile["cpv_codes"])
    print("Text:", zeile["text"][:1500])


# Nicht ausgewählte IT-Kandidaten
nicht_it_df = df[~df["it_kandidat_praezise"]].copy()

# Stichprobe mit gültigen CPV-Codes
stichprobe_mit_cpv = nicht_it_df[
    nicht_it_df["cpv_anzahl"] > 0
].sample(n=10, random_state=42)

# Stichprobe ohne gültige CPV-Codes
stichprobe_ohne_cpv = nicht_it_df[
    nicht_it_df["cpv_anzahl"] == 0
].sample(n=10, random_state=42)

# Beide Stichproben zusammenführen
negative_stichprobe = pd.concat([
    stichprobe_mit_cpv,
    stichprobe_ohne_cpv
])

print("\nStichprobe der nicht ausgewählten Kandidaten:")

for _, zeile in negative_stichprobe.iterrows():
    print("\n----------------------------")
    print("Dateiname:", zeile["filename"])
    print("CPV-Codes:", zeile["cpv_codes"])
    print("Text:", zeile["text"][:1500])


# Eindeutige Bekanntmachungs-ID erstellen
df["notice_id"] = df["filename"].str.extract(
    r"(\d+-\d{4})\.xml$"
)

# Vorläufige Trainingslabels erstellen
df["weak_label"] = df["it_kandidat_praezise"].astype(int)

# Aufbereiteten Datensatz speichern
df.to_csv(
    "ausschreibungen_2023_aufbereitet.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAufbereiteter Datensatz gespeichert!")

print("\nAnzahl der Datensätze:", len(df))

print("\nVorläufige Labels:")
print(df["weak_label"].value_counts())

print("\nSpalten des aufbereiteten Datensatzes:")
print(df.columns.tolist())
