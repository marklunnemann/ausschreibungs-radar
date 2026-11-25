
import pandas as pd
import re

# ==========================================
# 1. Konfiguration
# ==========================================

MAX_DATENSAETZE = 10000

EINGABEDATEI = (
    f"ausschreibungen_2023_{MAX_DATENSAETZE}_aufbereitet.csv"
)

AUSGABEDATEI = (
    f"ausschreibungen_2023_{MAX_DATENSAETZE}_nlp.csv"
)

CPV_MUSTER = r"\b\d{8}(?:-\d)?\b"

# ==========================================
# 2. Datensatz laden
# ==========================================

df = pd.read_csv(EINGABEDATEI)

df["text"] = df["text"].fillna("")

print("Geladene Datensätze:", len(df))

print("\nTextlängen vor der Bereinigung:")
print(df["text"].str.len().describe().round(2))

print("\nBeispiel eines unbearbeiteten Textes:")
print(df["text"].iloc[0][:2000])

# ==========================================
# 3. Funktion zur Textbereinigung
# ==========================================

def bereinige_text(text):

    # URLs entfernen
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # E-Mail-Adressen entfernen
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.\w+\b",
        " ",
        text
    )

    # CPV-Codes und mögliche CPV-Kandidaten entfernen
    text = re.sub(CPV_MUSTER, " ", text)

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

# ==========================================
# 4. Textbereinigung durchführen
# ==========================================

print("\nStarte Textbereinigung...")

df["text_clean"] = df["text"].apply(bereinige_text)

df["text_clean_length"] = df["text_clean"].str.len()

print("\nTextlängen nach der Bereinigung:")
print(df["text_clean_length"].describe().round(2))

print("\nBeispiel eines bereinigten Textes:")
print(df["text_clean"].iloc[0][:2000])

# ==========================================
# 5. Qualitätskontrolle
# ==========================================

cpv_vorher = df["text"].str.contains(
    CPV_MUSTER, regex=True
).sum()

cpv_nachher = df["text_clean"].str.contains(
    CPV_MUSTER, regex=True
).sum()

leere_texte = df["text_clean"].eq("").sum()

print("\nQualitätskontrolle:")

print("Texte mit CPV-Kandidaten vorher:", cpv_vorher)
print("Texte mit CPV-Kandidaten nachher:", cpv_nachher)
print("Leere bereinigte Texte:", leere_texte)

# Durchschnittliche Textlänge vergleichen
mittelwert_vorher = df["text"].str.len().mean()
mittelwert_nachher = df["text_clean_length"].mean()

reduktion = (
    1 - mittelwert_nachher / mittelwert_vorher
) * 100

print("\nDurchschnittliche Textlänge vorher:",
      round(mittelwert_vorher, 2))

print("Durchschnittliche Textlänge nachher:",
      round(mittelwert_nachher, 2))

print("Textreduktion in Prozent:",
      round(reduktion, 2))

# Die fünf längsten bereinigten Texte
print("\nDie fünf längsten bereinigten Texte:")

print(
    df.nlargest(5, "text_clean_length")[
        ["notice_id", "text_clean_length", "weak_label"]
    ].to_string(index=False)
)

# ==========================================
# 6. Leere Texte entfernen
# ==========================================

df = df[df["text_clean"].ne("")].copy()

# ==========================================
# 7. Duplikatprüfung
# ==========================================

print("\nDuplikatprüfung:")

print("Identische Originaltexte:",
      df["text"].duplicated().sum())

print("Identische bereinigte Texte:",
      df["text_clean"].duplicated().sum())

duplikate = df[
    df["text_clean"].duplicated(keep=False)
].copy()

print("Datensätze in Duplikatgruppen:",
      len(duplikate))

print("\nBeispiele identischer Dokumenttexte:")

print(
    duplikate[
        ["notice_id", "text_clean_length", "weak_label"]
    ].head(20).to_string(index=False)
)

# ==========================================
# 8. Widersprüchliche Labels untersuchen
# ==========================================

label_varianten = (
    duplikate.groupby("text_clean")["weak_label"]
    .nunique()
)

konflikt_texte = label_varianten[
    label_varianten > 1
].index

konflikte = df[
    df["text_clean"].isin(konflikt_texte)
].copy()

print("\nLabel-Konflikte:")

print("Duplikatgruppen mit widersprüchlichen Labels:",
      len(konflikt_texte))

print("Betroffene Datensätze:",
      len(konflikte))

if not konflikte.empty:

    print(
        konflikte[
            ["notice_id", "weak_label", "text_clean_length"]
        ].head(20).to_string(index=False)
    )

    # Konfliktfälle separat zur späteren Prüfung speichern
    konflikte.to_csv(
        f"label_konflikte_{MAX_DATENSAETZE}.csv",
        index=False,
        encoding="utf-8-sig"
    )

# ==========================================
# 9. Konflikte und Duplikate entfernen
# ==========================================

anzahl_vorher = len(df)

print("\nLabel-Verteilung vor der Duplikatentfernung:")
print(df["weak_label"].value_counts())

# Widersprüchlich gelabelte Texte ausschließen
df = df[
    ~df["text_clean"].isin(konflikt_texte)
].copy()

anzahl_nach_konflikten = len(df)

# Identische bereinigte Texte entfernen
df = df.drop_duplicates(
    subset=["text_clean"],
    keep="first"
).reset_index(drop=True)

print("\nErgebnisse der Duplikatentfernung:")

print("Datensätze vorher:", anzahl_vorher)

print("Entfernte Konfliktfälle:",
      anzahl_vorher - anzahl_nach_konflikten)

print("Entfernte weitere Duplikate:",
      anzahl_nach_konflikten - len(df))

print("Datensätze nachher:", len(df))

print("Insgesamt entfernte Datensätze:",
      anzahl_vorher - len(df))

# ==========================================
# 10. Finale Datenqualität
# ==========================================

print("\nFinale Qualitätskontrolle:")

print("Verbleibende leere Texte:",
      df["text_clean"].eq("").sum())

print("Verbleibende identische Texte:",
      df["text_clean"].duplicated().sum())

print("Verbleibende CPV-Kandidaten:",
      df["text_clean"].str.contains(
          CPV_MUSTER, regex=True
      ).sum())

print("\nFinale Verteilung der Trainingslabels:")
print(df["weak_label"].value_counts())

print("\nProzentuale Label-Verteilung:")

print(
    df["weak_label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ==========================================
# 11. Datenexport
# ==========================================

df.to_csv(
    AUSGABEDATEI,
    index=False,
    encoding="utf-8-sig"
)

print("\nNLP-Datensatz erfolgreich gespeichert!")

print("Dateiname:", AUSGABEDATEI)
print("Datensätze:", len(df))

print("\nSpalten des NLP-Datensatzes:")
print(df.columns.tolist())
