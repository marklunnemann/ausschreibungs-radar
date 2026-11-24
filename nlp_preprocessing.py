
import pandas as pd
import re

# Aufbereiteten Datensatz laden
df = pd.read_csv("ausschreibungen_2023_aufbereitet.csv")

# Datensatz überprüfen
print("Anzahl der Datensätze:", len(df))

print("\nTextlängen vor der Bereinigung:")
print(df["text"].str.len().describe().round(2))

print("\nBeispiel eines unbearbeiteten Textes:")
print(df["text"].iloc[0][:2000])



# Funktion zur Textbereinigung
def bereinige_text(text):

    # URLs entfernen
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # E-Mail-Adressen entfernen
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", text)

    # CPV-Codes und mögliche CPV-Kandidaten entfernen
    text = re.sub(r"\b\d{8}(?:-\d)?\b", " ", text)

    # Bekanntmachungs-IDs entfernen
    text = re.sub(
        r"Bekanntmachungs-ID:\s*\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Technische Identifikationsnummern entfernen
    text = re.sub(
        r"\b(?:LOT|ORG|PAR)-\d+\b",
        " ",
        text
    )

    # Technische Metadaten entfernen
    text = re.sub(
        r"(?m)^\s*(true|false|none|DEU|works|"
        r"services|supplies|ef-stand|sui-act|"
        r"tp-abil|anyw-cou)\s*$",
        " ",
        text
    )

    # Kleinschreibung
    text = text.lower()

    # Überflüssige Leerzeichen und Zeilenumbrüche
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Bereinigung auf alle Ausschreibungen anwenden
df["text_clean"] = df["text"].apply(bereinige_text)

# Ergebnisse überprüfen
print("\nTextlängen nach der Bereinigung:")
print(df["text_clean"].str.len().describe().round(2))

print("\nBeispiel eines bereinigten Textes:")
print(df["text_clean"].iloc[0][:2000])

print("\nLeere Texte nach der Bereinigung:")
print(df["text_clean"].eq("").sum())


# Qualitätskontrolle der Textbereinigung
cpv_muster = r"\b\d{8}(?:-\d)?\b"

cpv_vorher = df["text"].str.contains(
    cpv_muster, regex=True
).sum()

cpv_nachher = df["text_clean"].str.contains(
    cpv_muster, regex=True
).sum()

print("\nQualitätskontrolle:")
print("Texte mit CPV-Kandidaten vorher:", cpv_vorher)
print("Texte mit CPV-Kandidaten nachher:", cpv_nachher)

# Die fünf längsten bereinigten Texte untersuchen
df["text_clean_length"] = df["text_clean"].str.len()

print("\nDie fünf längsten bereinigten Texte:")

print(
    df.nlargest(5, "text_clean_length")[
        ["notice_id", "text_clean_length", "weak_label"]
    ].to_string(index=False)
)


# Identische Texte nach der Bereinigung prüfen
print("\nDoppelte bereinigte Texte:")
print(df["text_clean"].duplicated().sum())

# Die beiden längsten Texte vergleichen
text_1 = df.loc[
    df["notice_id"] == "687590-2023",
    "text_clean"
].iloc[0]

text_2 = df.loc[
    df["notice_id"] == "688525-2023",
    "text_clean"
].iloc[0]

print("\nVergleich der beiden längsten Texte:")
print("Texte identisch:", text_1 == text_2)

print("\nErste Bekanntmachung:")
print(text_1[:1000])

print("\nZweite Bekanntmachung:")
print(text_2[:1000])


# Anzahl vor der Duplikatentfernung
anzahl_vorher = len(df)

# Identische bereinigte Texte entfernen
df = df.drop_duplicates(
    subset=["text_clean"],
    keep="first"
).reset_index(drop=True)

print("\nDuplikatentfernung:")
print("Datensätze vorher:", anzahl_vorher)
print("Datensätze nachher:", len(df))
print("Entfernte Duplikate:", anzahl_vorher - len(df))

# Finalen NLP-Datensatz speichern
df.to_csv(
    "ausschreibungen_2023_nlp.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nNLP-Datensatz erfolgreich gespeichert!")

print("\nVerteilung der vorläufigen Trainingslabels:")
print(df["weak_label"].value_counts())

print("\nSpalten des NLP-Datensatzes:")
print(df.columns.tolist())
