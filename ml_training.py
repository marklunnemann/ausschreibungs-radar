
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Konfiguration
# ==========================================

RANDOM_STATE = 42

DATEI_KLEIN = "ausschreibungen_2023_nlp.csv"
DATEI_GROSS = "ausschreibungen_2023_10000_nlp.csv"

THRESHOLDS = [
    0.50, 0.45, 0.40, 0.35,
    0.30, 0.25, 0.20, 0.15, 0.10
]

# ==========================================
# 2. Datensätze laden
# ==========================================

df_klein = pd.read_csv(DATEI_KLEIN)
df_gross = pd.read_csv(DATEI_GROSS)

print("Kleiner Datensatz:", len(df_klein))
print("Großer Datensatz:", len(df_gross))

# ==========================================
# 3. Ursprünglichen Train-Test-Split
#    rekonstruieren
# ==========================================

train_klein, test_df = train_test_split(
    df_klein,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df_klein["weak_label"]
)

print("\nGemeinsamer Testdatensatz:")
print("Testdokumente:", len(test_df))
print(test_df["weak_label"].value_counts())

# Testdatensatz für spätere Evaluation sichern
test_df.to_csv(
    "ausschreibungen_testset_200.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 4. Data Leakage verhindern
# ==========================================

test_ids = set(test_df["notice_id"])
test_texte = set(test_df["text_clean"])

# Überschneidungen anhand der Bekanntmachungs-ID
id_treffer = df_gross["notice_id"].isin(test_ids)

# Überschneidungen anhand identischer Texte
text_treffer = df_gross["text_clean"].isin(test_texte)

# Beide Ausschlusskriterien kombinieren
ausschluss = id_treffer | text_treffer

train_gross = df_gross[~ausschluss].copy()

print("\nData-Leakage-Prüfung:")

print("Überschneidungen anhand der ID:",
      id_treffer.sum())

print("Überschneidungen anhand des Textes:",
      text_treffer.sum())

print("Insgesamt ausgeschlossene Dokumente:",
      ausschluss.sum())

# Sicherstellen, dass keine Überschneidungen verbleiben
assert not train_gross["notice_id"].isin(test_ids).any()
assert not train_gross["text_clean"].isin(test_texte).any()

assert not train_klein["notice_id"].isin(test_ids).any()
assert not train_klein["text_clean"].isin(test_texte).any()

print("Data-Leakage-Prüfung erfolgreich!")

# ==========================================
# 5. Trainingsdaten vergleichen
# ==========================================

print("\nExperiment 1 - Trainingsdaten:")
print("Anzahl:", len(train_klein))
print(train_klein["weak_label"].value_counts())

print("\nExperiment 2 - Trainingsdaten:")
print("Anzahl:", len(train_gross))
print(train_gross["weak_label"].value_counts())

X_test = test_df["text_clean"]
y_test = test_df["weak_label"]

# ==========================================
# 6. Evaluationsfunktion
# ==========================================

def evaluiere_modell(
    experiment,
    modellname,
    y_true,
    y_pred
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Experiment": experiment,
        "Modell": modellname,
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
        "Kandidaten": tp + fp
    }

def analysiere_thresholds(
    experiment,
    modellname,
    y_true,
    scores
):

    for threshold in THRESHOLDS:

        y_pred = (scores >= threshold).astype(int)

        ergebnis = evaluiere_modell(
            experiment,
            modellname,
            y_true,
            y_pred
        )

        ergebnis["Threshold"] = threshold

        threshold_ergebnisse.append(ergebnis)

# ==========================================
# 7. TF-IDF konfigurieren
# ==========================================

def erstelle_tfidf(optimiert=False):

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

    return TfidfVectorizer(**parameter)


# ==========================================
# 8. Modelle konfigurieren
# ==========================================

def erstelle_modelle():

    return {

        "M0 - Baseline": DummyClassifier(
            strategy="most_frequent"
        ),

        "M1 - Naive Bayes": MultinomialNB(
            alpha=1.0
        ),

        "M2 - Naive Bayes balanced": MultinomialNB(
            alpha=1.0,
            fit_prior=False
        ),

        "M3 - Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "M4 - Logistic Regression balanced":
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            )
    }


# ==========================================
# 9. Ergebnisse sammeln
# ==========================================

modell_ergebnisse = []
threshold_ergebnisse = []
feature_ergebnisse = []
vorhersage_ergebnisse = []

# ==========================================
# 10. Experiment durchführen
# ==========================================

def fuehre_experiment_durch(experiment, train_df):

    print("\n" + "=" * 60)
    print(experiment)
    print("=" * 60)

    X_train = train_df["text_clean"]
    y_train = train_df["weak_label"]

    # Standard-TF-IDF
    tfidf = erstelle_tfidf()

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("\nTF-IDF-Matrix Training:")
    print(X_train_tfidf.shape)

    print("\nTF-IDF-Matrix Test:")
    print(X_test_tfidf.shape)

    # Standardmodelle trainieren
    modelle = erstelle_modelle()

    for modellname, modell in modelle.items():

        print("\nTrainiere:", modellname)

        modell.fit(X_train_tfidf, y_train)

        y_pred = modell.predict(X_test_tfidf)

        ergebnis = evaluiere_modell(
            experiment,
            modellname,
            y_test,
            y_pred
        )

        modell_ergebnisse.append(ergebnis)

        print("Accuracy:", round(ergebnis["Accuracy"], 3))
        print("Precision:", round(ergebnis["Precision"], 3))
        print("Recall:", round(ergebnis["Recall"], 3))
        print("F1:", round(ergebnis["F1"], 3))


        # Threshold-Analyse für M4
        if modellname == "M4 - Logistic Regression balanced":

            scores_m4 = modell.predict_proba(
                X_test_tfidf
            )[:, 1]

            analysiere_thresholds(
                experiment,
                "M4",
                y_test,
                scores_m4
            )

            # Textmerkmale von M4 speichern
            feature_m4 = pd.DataFrame({
                "Experiment": experiment,
                "Modell": "M4",
                "Merkmal": tfidf.get_feature_names_out(),
                "Gewicht": modell.coef_[0]
            })

            feature_ergebnisse.append(feature_m4)

            # Einzelne Vorhersagen speichern
            vorhersagen_m4 = test_df[
                ["notice_id", "text", "cpv_codes", "weak_label"]
            ].copy()

            vorhersagen_m4["Experiment"] = experiment
            vorhersagen_m4["Modell"] = "M4"
            vorhersagen_m4["IT_Score"] = scores_m4

            vorhersagen_m4["Vorhersage_050"] = (
                scores_m4 >= 0.50
            ).astype(int)

            vorhersagen_m4["Vorhersage_030"] = (
                scores_m4 >= 0.30
            ).astype(int)

            vorhersage_ergebnisse.append(vorhersagen_m4)

    # ======================================
    # Optimierte TF-IDF-Vektorisierung
    # ======================================

    print("\nTrainiere M5 - Logistic Regression optimiert")

    tfidf_opt = erstelle_tfidf(optimiert=True)

    X_train_opt = tfidf_opt.fit_transform(X_train)
    X_test_opt = tfidf_opt.transform(X_test)

    lr_opt = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    lr_opt.fit(X_train_opt, y_train)

    y_pred_opt = lr_opt.predict(X_test_opt)

    modell_ergebnisse.append(
        evaluiere_modell(
            experiment,
            "M5 - Logistic Regression optimiert",
            y_test,
            y_pred_opt
        )
    )

    print("\nClassification Report M5:")

    print(
        classification_report(
            y_test,
            y_pred_opt,
            labels=[0, 1],
            target_names=[
                "Nicht ausgewählt",
                "Potenziell IT-relevant"
            ],
            zero_division=0
        )
    )

    print("\nConfusion Matrix M5:")

    print(
        confusion_matrix(
            y_test,
            y_pred_opt,
            labels=[0, 1]
        )
    )

    # ======================================
    # 11. Interpretierbarkeit
    # ======================================

    feature_df = pd.DataFrame({
        "Experiment": experiment,
        "Merkmal": tfidf_opt.get_feature_names_out(),
        "Gewicht": lr_opt.coef_[0]
    })

    feature_df["Modell"] = "M5"
    feature_ergebnisse.append(feature_df)

    print("\nTop 20 positive Textmerkmale:")

    print(
        feature_df.nlargest(20, "Gewicht")[
            ["Merkmal", "Gewicht"]
        ].to_string(index=False)
    )

    print("\nTop 20 negative Textmerkmale:")

    print(
        feature_df.nsmallest(20, "Gewicht")[
            ["Merkmal", "Gewicht"]
        ].to_string(index=False)
    )

    # ======================================
    # 12. Threshold-Analyse M5
    # ======================================

    scores = lr_opt.predict_proba(X_test_opt)[:, 1]

    analysiere_thresholds(
        experiment,
        "M5",
        y_test,
        scores
    )

    # ======================================
    # 13. Einzelne Vorhersagen speichern
    # ======================================

    vorhersagen = test_df[
        ["notice_id", "text", "cpv_codes", "weak_label"]
    ].copy()

    vorhersagen["Experiment"] = experiment
    vorhersagen["IT_Score"] = scores
    vorhersagen["Vorhersage_050"] = y_pred_opt

    vorhersagen["Vorhersage_030"] = (
        scores >= 0.30
    ).astype(int)

    vorhersage_ergebnisse.append(vorhersagen)
    vorhersagen["Modell"] = "M5"


# ==========================================
# 14. Beide Experimente ausführen
# ==========================================

fuehre_experiment_durch(
    "Experiment 1 - 1000",
    train_klein
)

fuehre_experiment_durch(
    "Experiment 2 - 10000",
    train_gross
)

# ==========================================
# 15. Modellvergleich
# ==========================================

modell_df = pd.DataFrame(modell_ergebnisse)

print("\n" + "=" * 60)
print("GESAMTER MODELLVERGLEICH")
print("=" * 60)

print(
    modell_df.round(3).to_string(index=False)
)


# ==========================================
# 16. Threshold-Vergleich
# ==========================================

threshold_df = pd.DataFrame(threshold_ergebnisse)

print("\nTHRESHOLD-ANALYSE:")

for experiment in threshold_df["Experiment"].unique():

    for modellname in ["M4", "M5"]:

        print("\n", experiment, "-", modellname)

        auswahl = threshold_df[
            (threshold_df["Experiment"] == experiment)
            & (threshold_df["Modell"] == modellname)
        ]

        print(
            auswahl[
                [
                    "Threshold",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "TP",
                    "FP",
                    "FN",
                    "TN",
                    "Kandidaten"
                ]
            ].round(3).to_string(index=False)
        )

# ==========================================
# 17. Direkter Leistungsvergleich
# ==========================================

print("\nVERGLEICH DER F1-SCORES:")

vergleich_f1 = modell_df.pivot(
    index="Modell",
    columns="Experiment",
    values="F1"
)

vergleich_f1["Differenz"] = (
    vergleich_f1["Experiment 2 - 10000"]
    - vergleich_f1["Experiment 1 - 1000"]
)

print(vergleich_f1.round(3).to_string())

# ==========================================
# 18. Ergebnisse exportieren
# ==========================================

modell_df.to_csv(
    "modellvergleich.csv",
    index=False,
    encoding="utf-8-sig"
)

threshold_df.to_csv(
    "threshold_vergleich.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.concat(feature_ergebnisse).to_csv(
    "feature_vergleich.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.concat(vorhersage_ergebnisse).to_csv(
    "vorhersage_vergleich.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAlle Experimente erfolgreich abgeschlossen!")

print("\nGespeicherte Ergebnisdateien:")

print("ausschreibungen_testset_200.csv")
print("modellvergleich.csv")
print("threshold_vergleich.csv")
print("feature_vergleich.csv")
print("vorhersage_vergleich.csv")
