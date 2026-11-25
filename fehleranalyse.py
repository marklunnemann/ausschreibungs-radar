
import pandas as pd

# Gespeicherte Modellvorhersagen laden
df = pd.read_csv("vorhersage_vergleich.csv")

# Experiment und Modell auswählen
EXPERIMENT = "Experiment 2 - 10000"
MODELL = "M5"
THRESHOLD = 0.40

df = df[
    (df["Experiment"] == EXPERIMENT)
    & (df["Modell"] == MODELL)
].copy()

# Vorhersagen anhand der Entscheidungsschwelle
df["vorhersage"] = (
    df["IT_Score"] >= THRESHOLD
).astype(int)


# Klassifikationstyp bestimmen
def klassifikationstyp(zeile):

    label = zeile["weak_label"]
    vorhersage = zeile["vorhersage"]

    if label == 1 and vorhersage == 1:
        return "True Positive"

    elif label == 0 and vorhersage == 1:
        return "False Positive"

    elif label == 1 and vorhersage == 0:
        return "False Negative"

    else:
        return "True Negative"


df["klassifikationstyp"] = df.apply(
    klassifikationstyp,
    axis=1
)

print("Experiment:", EXPERIMENT)
print("Modell:", MODELL)
print("Threshold:", THRESHOLD)

print("\nVerteilung der Klassifikationen:")
print(df["klassifikationstyp"].value_counts())


# ==========================================
# Qualitative Fehleranalyse
# ==========================================

for typ in [
    "False Positive",
    "False Negative",
    "True Positive"
]:

    auswahl = df[
        df["klassifikationstyp"] == typ
    ].sort_values(
        "IT_Score",
        ascending=False
    )

    print("\n" + "=" * 70)
    print(typ, "- Anzahl:", len(auswahl))
    print("=" * 70)

    for _, zeile in auswahl.iterrows():

        print("\n" + "-" * 50)

        print("Notice-ID:", zeile["notice_id"])
        print("CPV-Codes:", zeile["cpv_codes"])
        print("Weak Label:", zeile["weak_label"])
        print("Vorhersage:", zeile["vorhersage"])
        print("IT-Score:", round(zeile["IT_Score"], 4))

        print("\nAusschreibung:")
        print(str(zeile["text"])[:2500])


# ==========================================
# Datensatz für die manuelle Fehlerprüfung
# ==========================================

pruefung = df[
    df["klassifikationstyp"].isin([
        "False Positive",
        "False Negative",
        "True Positive"
    ])
].copy()

pruefung["manuelle_it_relevanz"] = ""
pruefung["unternehmensrelevanz"] = ""
pruefung["fehlerursache"] = ""
pruefung["begruendung"] = ""

pruefung.to_csv(
    "fehleranalyse_m5.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDatei gespeichert: fehleranalyse_m5.csv")
print("Zu untersuchende Dokumente:", len(pruefung))
