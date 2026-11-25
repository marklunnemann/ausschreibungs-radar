
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. Konfiguration
# ==========================================

RANDOM_STATE = 42

THRESHOLDS = [
    0.50, 0.45, 0.40, 0.35, 0.30,
    0.25, 0.20, 0.15, 0.10
]

# Angestrebter Recall auf den Validierungsdaten
ZIEL_RECALL = 0.90

# ==========================================
# 2. Trainingsdaten laden
# ==========================================

df_klein = pd.read_csv(
    "ausschreibungen_2023_nlp.csv"
)

df_gross = pd.read_csv(
    "ausschreibungen_2023_10000_nlp.csv"
)

# Ursprünglichen Testdatensatz rekonstruieren
_, alter_test = train_test_split(
    df_klein,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df_klein["weak_label"]
)

# Identische Trainingsgrundlage wie zuvor
train = df_gross[
    ~df_gross["notice_id"].isin(
        alter_test["notice_id"]
    )
    &
    ~df_gross["text_clean"].isin(
        alter_test["text_clean"]
    )
].copy()

X_train = train["text_clean"]
y_train = train["weak_label"]

print("Trainingsdokumente:", len(train))

print("\nTrainingslabels:")
print(y_train.value_counts())

# ==========================================
# 3. Fachliche Annotationen laden
# ==========================================

referenz = pd.read_csv(
    "validierung_referenz.csv"
)

annotation = pd.read_csv(
    "validierung_annotation_geprueft.csv",
    dtype={"manuelle_it_relevanz": str}
)

assert referenz["notice_id"].is_unique
assert annotation["notice_id"].is_unique

assert set(referenz["notice_id"]) == set(
    annotation["notice_id"]
)

df = referenz.merge(
    annotation[
        ["notice_id", "manuelle_it_relevanz"]
    ],
    on="notice_id",
    how="inner",
    validate="one_to_one"
)

df["manuelle_it_relevanz"] = (
    df["manuelle_it_relevanz"].str.strip()
)

print("\nAnnotationen:")
print(
    df["manuelle_it_relevanz"]
    .value_counts(dropna=False)
)

# Unklare Fälle separat speichern
unklar = df[
    ~df["manuelle_it_relevanz"].isin(["0", "1"])
].copy()

unklar.to_csv(
    "evaluation_unklare_faelle.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAusgeschlossene Fälle:", len(unklar))

# Nur binär annotierte Dokumente verwenden
df = df[
    df["manuelle_it_relevanz"].isin(["0", "1"])
].copy()

df["referenz_label"] = (
    df["manuelle_it_relevanz"].astype(int)
)

# ==========================================
# 4. Data-Leakage-Prüfung
# ==========================================

assert not df["notice_id"].isin(
    df_gross["notice_id"]
).any()

assert not df["text_clean"].isin(
    df_gross["text_clean"]
).any()

assert df["text_clean"].notna().all()
assert df["text_clean"].str.strip().ne("").all()

validierung = df[
    df["Stichprobe"] == "Validierung"
].copy()

test = df[
    df["Stichprobe"] == "Test"
].copy()

assert not validierung["notice_id"].isin(
    test["notice_id"]
).any()

assert not validierung["text_clean"].isin(
    test["text_clean"]
).any()

assert validierung["referenz_label"].nunique() == 2
assert test["referenz_label"].nunique() == 2

print("\nData-Leakage-Prüfung bestanden.")

print("\nValidierungsdokumente:", len(validierung))
print("Testdokumente:", len(test))

# ==========================================
# 5. Evaluationsfunktion
# ==========================================

def evaluiere(modell, threshold, y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Modell": modell,
        "Threshold": threshold,
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
        "Reduktion": 1 - (tp + fp) / len(y_true)
    }

# ==========================================
# 6. Modell-Pipelines definieren
# ==========================================

def erstelle_pipeline(optimiert=False):

    parameter = {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True
    }

    if optimiert:
        parameter["token_pattern"] = (
            r"(?u)\b(?=\w*[^\W\d_])\w{2,}\b"
        )

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(**parameter)
        ),
        (
            "klassifikator",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            )
        )
    ])


modelle = {
    "M4": erstelle_pipeline(optimiert=False),
    "M5": erstelle_pipeline(optimiert=True)
}

# ==========================================
# 7. Modelle trainieren
# ==========================================

for name, modell in modelle.items():

    print("\nTrainiere Modell:", name)

    modell.fit(X_train, y_train)

    print("Training abgeschlossen.")

# ==========================================
# 8. Evaluation auf Validierungsdaten
# ==========================================

X_val = validierung["text_clean"]
y_val = validierung["referenz_label"]

validierung_ergebnisse = []

# Regelbasierte CPV-Klassifikation
validierung_ergebnisse.append(
    evaluiere(
        "CPV",
        None,
        y_val,
        validierung["weak_label"]
    )
)

# Threshold-Analyse für M4 und M5
for name, modell in modelle.items():

    scores = modell.predict_proba(X_val)[:, 1]

    for threshold in THRESHOLDS:

        vorhersage = (
            scores >= threshold
        ).astype(int)

        ergebnis = evaluiere(
            name,
            threshold,
            y_val,
            vorhersage
        )

        validierung_ergebnisse.append(ergebnis)

val_df = pd.DataFrame(validierung_ergebnisse)

print("\n" + "=" * 70)
print("VALIDIERUNGSERGEBNISSE")
print("=" * 70)

print(
    val_df.round(3).to_string(index=False)
)

val_df.to_csv(
    "finale_validierung_ergebnisse.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 9. Modellauswahl
# ==========================================

ml_ergebnisse = val_df[
    val_df["Modell"].isin(["M4", "M5"])
].copy()

geeignet = ml_ergebnisse[
    ml_ergebnisse["Recall"] >= ZIEL_RECALL
].copy()

if not geeignet.empty:

    # Bei erreichtem Recall-Ziel:
    # höchste Precision bevorzugen
    bestes = geeignet.sort_values(
        ["Precision", "F1", "Threshold"],
        ascending=False
    ).iloc[0]

    auswahlgrund = (
        "Hoechste Precision bei erreichtem Recall-Ziel"
    )

else:

    # Falls Recall-Ziel nicht erreichbar:
    # besten F1-Score auswählen
    bestes = ml_ergebnisse.sort_values(
        ["F1", "Recall", "Precision"],
        ascending=False
    ).iloc[0]

    auswahlgrund = (
        "Bester F1-Score, da Recall-Ziel nicht erreicht"
    )

finaler_name = bestes["Modell"]
finaler_threshold = float(bestes["Threshold"])

finales_modell = modelle[finaler_name]

print("\n" + "=" * 70)
print("MODELLAUSWAHL")
print("=" * 70)

print("Ausgewähltes Modell:", finaler_name)
print("Threshold:", finaler_threshold)
print("Auswahlgrund:", auswahlgrund)

print("\nValidierungsleistung:")
print(bestes.to_string())

# ==========================================
# 10. Abschließende Testevaluation
# ==========================================

X_test = test["text_clean"]
y_test = test["referenz_label"]

print("\nTestlabel-Verteilung:")
print(y_test.value_counts())

test_ergebnisse = []

# CPV als Vergleichsverfahren
test_ergebnisse.append(
    evaluiere(
        "CPV",
        None,
        y_test,
        test["weak_label"]
    )
)

# Ausschließlich das zuvor ausgewählte ML-Modell
test_scores = finales_modell.predict_proba(
    X_test
)[:, 1]

test_vorhersage = (
    test_scores >= finaler_threshold
).astype(int)

test_ergebnisse.append(
    evaluiere(
        finaler_name,
        finaler_threshold,
        y_test,
        test_vorhersage
    )
)

test_df = pd.DataFrame(test_ergebnisse)

print("\n" + "=" * 70)
print("UNABHÄNGIGE TESTEVALUATION")
print("=" * 70)

print(
    test_df.round(3).to_string(index=False)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_vorhersage,
        labels=[0, 1],
        target_names=[
            "Nicht IT-relevant",
            "IT-relevant"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_vorhersage,
        labels=[0, 1]
    )
)

# ==========================================
# 11. Testergebnisse speichern
# ==========================================

test_df.to_csv(
    "finale_test_ergebnisse.csv",
    index=False,
    encoding="utf-8-sig"
)

test["IT_Score"] = test_scores
test["ML_Vorhersage"] = test_vorhersage

test.to_csv(
    "finale_test_vorhersagen.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 12. Finales Modell speichern
# ==========================================

modellpaket = {
    "pipeline": finales_modell,
    "modellname": finaler_name,
    "threshold": finaler_threshold,
    "random_state": RANDOM_STATE,
    "trainingsdokumente": len(train),
    "eingabeformat": "text_clean",
    "auswahlgrund": auswahlgrund
}

joblib.dump(
    modellpaket,
    "ausschreibungsradar_nlp_modell.joblib"
)

print("\n" + "=" * 70)
print("EVALUATION ABGESCHLOSSEN")
print("=" * 70)

print("\nGespeicherte Dateien:")
print("finale_validierung_ergebnisse.csv")
print("finale_test_ergebnisse.csv")
print("finale_test_vorhersagen.csv")
print("evaluation_unklare_faelle.csv")
print("ausschreibungsradar_nlp_modell.joblib")
