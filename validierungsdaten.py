
import pandas as pd
import re
import random

from datasets import load_dataset

# ==========================================
# 1. Konfiguration
# ==========================================

RANDOM_STATE = 42
SKIP_DATENSAETZE = 10000
KANDIDATENPOOL = 5000
STICHPROBE = 200

# ==========================================
# 2. Bisherige Daten laden
# ==========================================

bisher = pd.read_csv(
    "ausschreibungen_2023_10000.csv"
)

bekannte_dateien = set(bisher["filename"])

print("Bisherige Dokumente:", len(bisher))

# ==========================================
# 3. Neue Bekanntmachungen laden
# ==========================================

dataset = load_dataset(
    "PleIAs/TEDEUTenders",
    split="train",
    streaming=True
)

neue_daten = []
deutsche_dokumente = 0

for eintrag in dataset:

    if eintrag.get("language") != "DEU":
        continue

    deutsche_dokumente += 1

    # Bisherigen Datenbestand überspringen
    if deutsche_dokumente <= SKIP_DATENSAETZE:
        continue

    if eintrag.get("filename") in bekannte_dateien:
        continue

    neue_daten.append(eintrag)

    if len(neue_daten) % 500 == 0:
        print(
            "Neue Kandidaten:",
            len(neue_daten)
        )

    if len(neue_daten) >= KANDIDATENPOOL:
        break

df = pd.DataFrame(neue_daten)

print("\nNeue Dokumente:", len(df))

if df.empty:
    raise ValueError("Keine neuen Dokumente gefunden.")

print("\nVerfügbare Spalten:")
print(df.columns.tolist())

# ==========================================
# 4. Erste Duplikatprüfung
# ==========================================

df = df.drop_duplicates(
    subset=["filename"]
).copy()

df = df.dropna(subset=["text"])

df = df[
    ~df["text"].isin(bisher["text"])
].copy()

df = df.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)

print("\nKandidaten nach Duplikatprüfung:", len(df))

# ==========================================
# 5. Kandidatenpool speichern
# ==========================================

df.to_csv(
    "validierung_kandidatenpool.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nKandidatenpool erfolgreich gespeichert.")
