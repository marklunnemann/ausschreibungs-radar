
import pandas as pd
import re

# ==========================================
# 1. Konfiguration
# ==========================================

RANDOM_STATE = 42

DATEI_NEU = "validierung_kandidatenpool.csv"
DATEI_ALT = "ausschreibungen_2023_10000_aufbereitet.csv"
DATEI_NLP = "ausschreibungen_2023_10000_nlp.csv"
DATEI_CPV = "cpv_2008_ver_2013.xlsx"

CPV_MUSTER = r"\b\d{8}(?:-\d)?\b"

IT_PREFIXE = ("302", "324", "48", "503", "72")

# ==========================================
# 2. Datensätze laden
# ==========================================

df = pd.read_csv(DATEI_NEU)
bisher = pd.read_csv(DATEI_ALT)
bisher_nlp = pd.read_csv(DATEI_NLP)

df["text"] = df["text"].fillna("")

print("Neue Kandidaten:", len(df))

# ==========================================
# 3. CPV-Referenzverzeichnis
# ==========================================

cpv = pd.read_excel(DATEI_CPV)

gueltige_codes = set(
    cpv["CODE"]
    .astype(str)
    .str.extract(r"^(\d{8})-\d$")[0]
    .dropna()
)

print("Gültige CPV-Codes:", len(gueltige_codes))


def extrahiere_cpv(text):

    kandidaten = re.findall(CPV_MUSTER, text)

    return sorted({
        code[:8]
        for code in kandidaten
        if code[:8] in gueltige_codes
    })


df["cpv_codes"] = df["text"].apply(extrahiere_cpv)

df["weak_label"] = df["cpv_codes"].apply(
    lambda codes: int(
        any(code.startswith(IT_PREFIXE) for code in codes)
    )
)

# ==========================================
# 4. Text-Preprocessing
# ==========================================

def bereinige_text(text):

    text = re.sub(
        r"https?://\S+|www\.\S+", " ", text
    )

    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", text
    )

    text = re.sub(CPV_MUSTER, " ", text)

    text = re.sub(
        r"Bekanntmachungs-ID:\s*\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\b(?:LOT|ORG|PAR)-\d+\b", " ", text
    )

    text = re.sub(
        r"(?m)^\s*(true|false|none|DEU|works|"
        r"services|supplies|ef-stand|sui-act|"
        r"tp-abil|anyw-cou)\s*$",
        " ",
        text
    )

    text = text.lower()

    return re.sub(r"\s+", " ", text).strip()


df["text_clean"] = df["text"].apply(bereinige_text)

df["notice_id"] = df["filename"].str.extract(
    r"(\d+-\d{4})\.xml$"
)[0]

# ==========================================
# 5. Data-Leakage-Prüfung
# ==========================================

anzahl_vorher = len(df)

df = df.dropna(subset=["notice_id"])

df = df[df["text_clean"].ne("")].copy()

# Abgleich mit sämtlichen bisherigen IDs
df = df[
    ~df["notice_id"].isin(bisher["notice_id"])
].copy()

# Abgleich mit sämtlichen bisherigen Texten
bekannte_texte = set(
    bisher["text"].fillna("").apply(bereinige_text)
)

bekannte_texte.update(bisher_nlp["text_clean"].dropna())

df = df[
    ~df["text_clean"].isin(bekannte_texte)
].copy()

# Widersprüchliche Labels bei identischen Texten
label_varianten = df.groupby(
    "text_clean"
)["weak_label"].nunique()

konflikte = label_varianten[label_varianten > 1].index

df = df[
    ~df["text_clean"].isin(konflikte)
].copy()

# Interne Duplikate entfernen
df = df.drop_duplicates(subset=["notice_id"])
df = df.drop_duplicates(subset=["text_clean"])
df = df.reset_index(drop=True)

assert not df["notice_id"].isin(
    bisher["notice_id"]
).any()

assert not df["text_clean"].isin(
    bekannte_texte
).any()

print("\nData-Leakage-Prüfung:")
print("Dokumente vorher:", anzahl_vorher)
print("Dokumente nachher:", len(df))
print("Entfernte Dokumente:", anzahl_vorher - len(df))

print("\nWeak-Label-Verteilung:")
print(df["weak_label"].value_counts())

print("\nProzentuale Verteilung:")
print(
    df["weak_label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ==========================================
# 6. Unabhängige Zufallsstichprobe
# ==========================================

test = df.sample(
    n=100,
    random_state=RANDOM_STATE
).copy()

test["Stichprobe"] = "Test"

# Testdaten aus dem Kandidatenpool entfernen
rest = df[
    ~df["notice_id"].isin(test["notice_id"])
].copy()

# ==========================================
# 7. Angereicherte Validierungsstichprobe
# ==========================================

positiv = rest[
    rest["weak_label"] == 1
].sample(
    n=50,
    random_state=RANDOM_STATE
)

negativ = rest[
    rest["weak_label"] == 0
].sample(
    n=50,
    random_state=RANDOM_STATE
)

validierung = pd.concat(
    [positiv, negativ],
    ignore_index=True
)

validierung["Stichprobe"] = "Validierung"

assert not validierung["notice_id"].isin(
    test["notice_id"]
).any()

# ==========================================
# 8. Gemeinsamen Datensatz erstellen
# ==========================================

gesamt = pd.concat(
    [test, validierung],
    ignore_index=True
)

gesamt = gesamt.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("\nStichprobenverteilung:")

print(
    pd.crosstab(
        gesamt["Stichprobe"],
        gesamt["weak_label"]
    )
)

# Vollständigen Referenzdatensatz speichern
gesamt.to_csv(
    "validierung_referenz.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 9. Verblindete Annotation vorbereiten
# ==========================================

# Bereinigte Texte ohne extrahierte CPV-Codes
annotation = gesamt[
    ["notice_id", "text_clean"]
].copy()

annotation["manuelle_it_relevanz"] = ""
annotation["it_leistungsanteil"] = ""
annotation["leistungsbereich"] = ""
annotation["begruendung"] = ""

annotation.to_csv(
    "validierung_annotation.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDateien erfolgreich gespeichert:")
print("validierung_referenz.csv")
print("validierung_annotation.csv")

print("\nZu bewertende Dokumente:", len(annotation))
