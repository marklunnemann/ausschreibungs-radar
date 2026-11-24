
from datasets import load_dataset

# Nur die ersten 100 deutschen Ausschreibungen laden
daten = load_dataset(
    "PleIAs/TEDEUTenders",
    split="train",
    streaming=True
)

deutsche_daten = []

for eintrag in daten:
    if eintrag["language"] == "DEU":
        deutsche_daten.append(eintrag)

    if len(deutsche_daten) == 100:
        break

print(f"{len(deutsche_daten)} deutsche Ausschreibungen gefunden")

# Erste Ausschreibung anzeigen
print(deutsche_daten[0]["text"][:1000])


import pandas as pd

# Ausschreibungen in eine Tabelle umwandeln
df = pd.DataFrame(deutsche_daten)

# Als CSV-Datei speichern
df.to_csv(
    "ausschreibungen_de.csv",
    index=False,
    encoding="utf-8-sig"
)

print("CSV-Datei erfolgreich gespeichert!")
print("Spalten:", df.columns.tolist())

print("\nDatenübersicht:")
print(df.info())

print("\nFehlende Werte:")
print(df.isnull().sum())

print("\nDoppelte Ausschreibungen:")
print(df.duplicated(subset=["identifier"]).sum())


import re

# CPV-Codes in Ausschreibungstexten suchen
# Beispiel: 72000000-5
cpv_muster = r"\b\d{8}-\d\b"

anzahl_mit_cpv = 0

for text in df["text"]:
    cpv_codes = re.findall(cpv_muster, text)

    if cpv_codes:
        anzahl_mit_cpv += 1
        print("Gefundene CPV-Codes:", cpv_codes[:5])

print("\nErgebnis der CPV-Prüfung:")
print(f"Ausschreibungen mit CPV-Code: {anzahl_mit_cpv}")
print(f"Ausschreibungen ohne CPV-Code: {len(df) - anzahl_mit_cpv}")



# Prüfen, ob das Wort CPV im Text vorkommt
anzahl_cpv = df["text"].str.contains(
    "CPV", case=False, na=False
).sum()

print(f"Texte mit dem Begriff CPV: {anzahl_cpv}")

# Beispieltext mit CPV anzeigen
treffer = df[
    df["text"].str.contains("CPV", case=False, na=False)
]

if not treffer.empty:
    text = treffer.iloc[0]["text"]
    position = text.lower().find("cpv")
    print(text[max(0, position-100):position+300])



# Verteilung der Ausschreibungen nach Jahren
df["jahr"] = pd.to_datetime(
    df["date"],
    format="%d/%m/%Y",
    errors="coerce"
).dt.year

print("\nAusschreibungen nach Jahr:")
print(df["jahr"].value_counts().sort_index())


print("\nBeispiele für Datumsangaben:")
print(df["date"].head(15).to_string(index=False))

print("\nNicht erkannte Datumsangaben:")
print(
    df.loc[df["jahr"].isna(), "date"]
    .head(10)
    .to_string(index=False)
)
