
from datasets import load_dataset
import re


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

    if len(deutsche_daten) >= 1000:
        break

print(f"{len(deutsche_daten)} deutsche Ausschreibungen gefunden")

if deutsche_daten:
    print("\nErste Ausschreibung:")
    print(deutsche_daten[0]["text"][:500])

anzahl_cpv = 0

for eintrag in deutsche_daten:
    text = eintrag["text"]

    # CPV-Codes mit oder ohne Prüfziffer suchen
    codes = re.findall(r"\b\d{8}(?:-\d)?\b", text)

    if codes:
        anzahl_cpv += 1
        print("Gefundene Codes:", codes[:5])

print("\nCPV-Prüfung 2023:")
print(f"Texte mit möglichen CPV-Codes: {anzahl_cpv}")
print(f"Texte ohne Treffer: {len(deutsche_daten) - anzahl_cpv}")


# Für unser IT-Systemhaus relevante CPV-Abteilungen
it_bereiche = ("30", "32", "48", "50", "72")

it_ausschreibungen = 0

for eintrag in deutsche_daten:
    codes = re.findall(
        r"\b\d{8}(?:-\d)?\b",
        eintrag["text"]
    )

    if any(code.startswith(it_bereiche) for code in codes):
        it_ausschreibungen += 1

print("\nErste IT-Klassifikation:")
print(f"Potenziell IT-relevant: {it_ausschreibungen}")
print(f"Nicht als IT-relevant erkannt: {len(deutsche_daten) - it_ausschreibungen}")


print("\nPotenziell relevante IT-Ausschreibungen:")

for eintrag in deutsche_daten:
    codes = re.findall(
        r"\b\d{8}(?:-\d)?\b",
        eintrag["text"]
    )

    it_codes = [
        code for code in codes
        if code.startswith(it_bereiche)
    ]

    if it_codes:
        print("\n----------------------------")
        print("CPV-Kandidaten:", sorted(set(it_codes)))
        print("Ausschreibung:")
        print(eintrag["text"][:300])


import pandas as pd

# Ausschreibungen in DataFrame umwandeln
df = pd.DataFrame(deutsche_daten)

# Potenzielle IT-Relevanz bestimmen
def ist_it_relevant(text):
    codes = re.findall(r"\b\d{8}(?:-\d)?\b", text)
    return any(code.startswith(it_bereiche) for code in codes)

df["it_kandidat"] = df["text"].apply(ist_it_relevant)

# Ergebnisse speichern
df.to_csv(
    "ausschreibungen_2023_1000.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nCSV-Datei gespeichert!")
print(df["it_kandidat"].value_counts())
