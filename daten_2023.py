
from datasets import load_dataset
import pandas as pd
import re

# ==========================================
# 1. Konfiguration
# ==========================================

MAX_DATENSAETZE = 10000

IT_BEREICHE = ("30", "32", "48", "50", "72")

CPV_MUSTER = re.compile(r"\b\d{8}(?:-\d)?\b")

# ==========================================
# 2. Datenbeschaffung
# ==========================================

print("Lade deutsche Ausschreibungen aus 2023...")

daten = load_dataset(
    "PleIAs/TEDEUTenders",
    data_files="2023_EUTenders.parquet",
    split="train",
    streaming=True
)

deutsche_daten = []

for eintrag in daten:

    if eintrag["language"] == "DEU":
        deutsche_daten.append(eintrag)

        if len(deutsche_daten) % 1000 == 0:
            print(
                f"Fortschritt: {len(deutsche_daten)}"
                f" / {MAX_DATENSAETZE}"
            )

    if len(deutsche_daten) >= MAX_DATENSAETZE:
        break

print(f"\n{len(deutsche_daten)} deutsche Ausschreibungen gefunden")

# ==========================================
# 3. DataFrame erstellen
# ==========================================

df = pd.DataFrame(deutsche_daten)

df["text"] = df["text"].fillna("")

print("\nErste Ausschreibung:")
print(df["text"].iloc[0][:500])

# ==========================================
# 4. CPV-Kandidaten extrahieren
# ==========================================

cpv_kandidaten = df["text"].apply(
    lambda text: sorted(set(CPV_MUSTER.findall(text)))
)

anzahl_cpv = cpv_kandidaten.apply(bool).sum()

print("\nCPV-Prüfung 2023:")
print("Texte mit möglichen CPV-Codes:", anzahl_cpv)
print("Texte ohne Treffer:", len(df) - anzahl_cpv)

print(
    "CPV-Erkennungsquote:",
    round(anzahl_cpv / len(df) * 100, 2),
    "%"
)

# ==========================================
# 5. Vorläufige IT-Klassifikation
# ==========================================

df["it_kandidat"] = cpv_kandidaten.apply(
    lambda codes: any(
        code.startswith(IT_BEREICHE)
        for code in codes
    )
)

it_anzahl = df["it_kandidat"].sum()

print("\nErste IT-Klassifikation:")
print("Potenziell IT-relevant:", it_anzahl)
print("Nicht ausgewählt:", len(df) - it_anzahl)

print(
    "Anteil potenziell IT-relevant:",
    round(it_anzahl / len(df) * 100, 2),
    "%"
)

# ==========================================
# 6. Explorative Datenanalyse
# ==========================================

print("\nTextlängen in Zeichen:")
print(df["text"].str.len().describe().round(2))

print("\nKlassenverteilung:")
print(df["it_kandidat"].value_counts())

print("\nProzentuale Klassenverteilung:")
print(
    df["it_kandidat"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ==========================================
# 7. Beispielhafte IT-Kandidaten
# ==========================================

print("\nErste zehn potenzielle IT-Ausschreibungen:")

for index, zeile in df[df["it_kandidat"]].head(10).iterrows():

    it_codes = [
        code for code in cpv_kandidaten.loc[index]
        if code.startswith(IT_BEREICHE)
    ]

    print("\n----------------------------")
    print("CPV-Kandidaten:", it_codes)
    print("Ausschreibung:")
    print(zeile["text"][:300])

# ==========================================
# 8. Datenexport
# ==========================================

dateiname = f"ausschreibungen_2023_{MAX_DATENSAETZE}.csv"

df.to_csv(
    dateiname,
    index=False,
    encoding="utf-8-sig"
)

print("\nCSV-Datei erfolgreich gespeichert!")
print("Dateiname:", dateiname)
print("Datensätze:", len(df))
print("Spalten:", df.columns.tolist())
