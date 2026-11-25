
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# 1. Testergebnisse laden
# ==========================================

df = pd.read_csv("finale_test_vorhersagen.csv")

y_true = df["referenz_label"].astype(int)

print("Testdokumente:", len(df))

print("\nReferenzlabels:")
print(y_true.value_counts())

# ==========================================
# 2. Klassifikationsstrategien
# ==========================================

df["CPV"] = df["weak_label"].astype(int)

df["ML"] = df["ML_Vorhersage"].astype(int)

df["Hybrid_OR"] = (
    (df["CPV"] == 1) |
    (df["ML"] == 1)
).astype(int)

df["Hybrid_AND"] = (
    (df["CPV"] == 1) &
    (df["ML"] == 1)
).astype(int)

# ==========================================
# 3. Evaluationsfunktion
# ==========================================

def evaluiere(name, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Strategie": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Kandidaten": tp + fp,
        "Reduktion": 1 - (tp + fp) / len(df)
    }

# ==========================================
# 4. Strategien vergleichen
# ==========================================

strategien = [
    "CPV",
    "ML",
    "Hybrid_OR",
    "Hybrid_AND"
]

ergebnisse = []

for strategie in strategien:

    ergebnisse.append(
        evaluiere(
            strategie,
            df[strategie]
        )
    )

ergebnis_df = pd.DataFrame(ergebnisse)

print("\n" + "=" * 70)
print("VERGLEICH DER KLASSIFIKATIONSSTRATEGIEN")
print("=" * 70)

print(
    ergebnis_df.round(3).to_string(index=False)
)

# ==========================================
# 5. Überschneidungen untersuchen
# ==========================================

print("\nÜberschneidung CPV und ML:")

print(
    pd.crosstab(
        df["CPV"],
        df["ML"],
        rownames=["CPV"],
        colnames=["ML"]
    )
)

relevant = df[df["referenz_label"] == 1]

print("\nÜberschneidung bei relevanten Dokumenten:")

print(
    pd.crosstab(
        relevant["CPV"],
        relevant["ML"],
        rownames=["CPV"],
        colnames=["ML"]
    )
)

# ==========================================
# 6. Exklusive Treffer identifizieren
# ==========================================

nur_cpv = relevant[
    (relevant["CPV"] == 1) &
    (relevant["ML"] == 0)
]

nur_ml = relevant[
    (relevant["CPV"] == 0) &
    (relevant["ML"] == 1)
]

beide = relevant[
    (relevant["CPV"] == 1) &
    (relevant["ML"] == 1)
]

keines = relevant[
    (relevant["CPV"] == 0) &
    (relevant["ML"] == 0)
]

print("\nRelevante Ausschreibungen:")

print("Nur CPV erkannt:", len(nur_cpv))
print("Nur ML erkannt:", len(nur_ml))
print("Beide erkannt:", len(beide))
print("Keines erkannt:", len(keines))

# ==========================================
# 7. Qualitative Analyse
# ==========================================

gruppen = {
    "Nur CPV erkannt": nur_cpv,
    "Nur ML erkannt": nur_ml,
    "Von beiden übersehen": keines
}

for name, gruppe in gruppen.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for _, zeile in gruppe.iterrows():

        print("\nNotice-ID:", zeile["notice_id"])
        print("CPV-Codes:", zeile["cpv_codes"])
        print("IT-Score:", round(zeile["IT_Score"], 4))

        print("\nAusschreibung:")
        print(str(zeile["text"])[:2000])

# ==========================================
# 8. Ergebnisse speichern
# ==========================================

ergebnis_df.to_csv(
    "hybrid_evaluation_ergebnisse.csv",
    index=False,
    encoding="utf-8-sig"
)

df.to_csv(
    "hybrid_evaluation_vorhersagen.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDateien erfolgreich gespeichert:")
print("hybrid_evaluation_ergebnisse.csv")
print("hybrid_evaluation_vorhersagen.csv")
