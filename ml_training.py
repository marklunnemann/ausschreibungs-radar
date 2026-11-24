
import pandas as pd
from sklearn.model_selection import train_test_split

# NLP-Datensatz laden
df = pd.read_csv("ausschreibungen_2023_nlp.csv")

# Eingabedaten und Zielvariable definieren
X = df["text_clean"]
y = df["weak_label"]

# Trainings- und Testdatensatz erstellen
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Ergebnisse überprüfen
print("Datensätze insgesamt:", len(df))

print("\nTrainingsdatensatz:", len(X_train))
print("Testdatensatz:", len(X_test))

print("\nKlassenverteilung im Trainingsdatensatz:")
print(y_train.value_counts())

print("\nKlassenverteilung im Testdatensatz:")
print(y_test.value_counts())

print("\nProzentuale Verteilung im Trainingsdatensatz:")
print(y_train.value_counts(normalize=True).mul(100).round(2))

print("\nProzentuale Verteilung im Testdatensatz:")
print(y_test.value_counts(normalize=True).mul(100).round(2))


from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF-Vektorisierung konfigurieren
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# TF-IDF ausschließlich auf Trainingsdaten anlernen
X_train_tfidf = tfidf.fit_transform(X_train)

# Testdaten mit demselben Vokabular transformieren
X_test_tfidf = tfidf.transform(X_test)

# Ergebnisse überprüfen
print("\nTF-IDF-Vektorisierung erfolgreich!")

print("\nDimensionen der Trainingsmatrix:")
print(X_train_tfidf.shape)

print("\nDimensionen der Testmatrix:")
print(X_test_tfidf.shape)

print("\nAnzahl der Textmerkmale:")
print(len(tfidf.get_feature_names_out()))

print("\nErste 30 Textmerkmale:")
print(tfidf.get_feature_names_out()[:30])


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Naive-Bayes-Modell erstellen
nb_model = MultinomialNB(alpha=1.0)

# Modell trainieren
nb_model.fit(X_train_tfidf, y_train)

# Vorhersagen für den Testdatensatz erstellen
y_pred_nb = nb_model.predict(X_test_tfidf)

# Ergebnisse ausgeben
print("\nNaive Bayes erfolgreich trainiert!")

print("\nAccuracy:")
print(round(accuracy_score(y_test, y_pred_nb), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_nb,
        labels=[0, 1],
        target_names=[
            "Nicht ausgewählt",
            "Potenziell IT-relevant"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb, labels=[0, 1]))

print("\nVerteilung der vorhergesagten Klassen:")
print(pd.Series(y_pred_nb).value_counts())


# Naive Bayes mit gleichgewichteten Klassenprioren
nb_balanced = MultinomialNB(
    alpha=1.0,
    fit_prior=False
)

# Modell trainieren
nb_balanced.fit(X_train_tfidf, y_train)

# Vorhersagen erstellen
y_pred_nb_balanced = nb_balanced.predict(X_test_tfidf)

# Ergebnisse ausgeben
print("\nNaive Bayes mit gleichen Klassenprioren:")

print("\nAccuracy:")
print(round(
    accuracy_score(y_test, y_pred_nb_balanced),
    4
))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_nb_balanced,
        labels=[0, 1],
        target_names=[
            "Nicht ausgewählt",
            "Potenziell IT-relevant"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_nb_balanced,
        labels=[0, 1]
    )
)

print("\nVerteilung der vorhergesagten Klassen:")
print(pd.Series(y_pred_nb_balanced).value_counts())


from sklearn.linear_model import LogisticRegression

# Logistische Regression erstellen
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Modell trainieren
lr_model.fit(X_train_tfidf, y_train)

# Vorhersagen erstellen
y_pred_lr = lr_model.predict(X_test_tfidf)

# Ergebnisse ausgeben
print("\nLogistic Regression erfolgreich trainiert!")

print("\nAccuracy:")
print(round(accuracy_score(y_test, y_pred_lr), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_lr,
        labels=[0, 1],
        target_names=[
            "Nicht ausgewählt",
            "Potenziell IT-relevant"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr, labels=[0, 1]))

print("\nVerteilung der vorhergesagten Klassen:")
print(pd.Series(y_pred_lr).value_counts())


# Logistische Regression mit Klassengewichtung
lr_balanced = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

# Modell trainieren
lr_balanced.fit(X_train_tfidf, y_train)

# Vorhersagen erstellen
y_pred_lr_balanced = lr_balanced.predict(X_test_tfidf)

# Ergebnisse ausgeben
print("\nLogistic Regression mit Klassengewichtung:")

print("\nAccuracy:")
print(round(
    accuracy_score(y_test, y_pred_lr_balanced),
    4
))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_lr_balanced,
        labels=[0, 1],
        target_names=[
            "Nicht ausgewählt",
            "Potenziell IT-relevant"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_lr_balanced,
        labels=[0, 1]
    )
)

print("\nVerteilung der vorhergesagten Klassen:")
print(pd.Series(y_pred_lr_balanced).value_counts())


# Analyse der gelernten Textmerkmale
feature_names = tfidf.get_feature_names_out()

koeffizienten = lr_balanced.coef_[0]

feature_df = pd.DataFrame({
    "merkmal": feature_names,
    "gewicht": koeffizienten
})

# Merkmale, die für IT-Relevanz sprechen
print("\nTop 30 positive Textmerkmale:")

print(
    feature_df.nlargest(30, "gewicht")
    .to_string(index=False)
)

# Merkmale, die gegen IT-Relevanz sprechen
print("\nTop 30 negative Textmerkmale:")

print(
    feature_df.nsmallest(30, "gewicht")
    .to_string(index=False)
)


# Optimierte TF-IDF-Vektorisierung
tfidf_optimiert = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    token_pattern=r"(?u)\b(?=\w*[^\W\d_])\w{2,}\b"
)

# Trainings- und Testdaten transformieren
X_train_opt = tfidf_optimiert.fit_transform(X_train)
X_test_opt = tfidf_optimiert.transform(X_test)

# Gewichtete logistische Regression
lr_optimiert = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

# Modell trainieren
lr_optimiert.fit(X_train_opt, y_train)

# Vorhersagen erstellen
y_pred_opt = lr_optimiert.predict(X_test_opt)

print("\nOptimierte Logistic Regression:")

print("\nAccuracy:")
print(round(accuracy_score(y_test, y_pred_opt), 4))

print("\nClassification Report:")
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

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_opt, labels=[0, 1]))

# Optimierte Textmerkmale untersuchen
feature_opt_df = pd.DataFrame({
    "merkmal": tfidf_optimiert.get_feature_names_out(),
    "gewicht": lr_optimiert.coef_[0]
})

print("\nTop 20 optimierte positive Textmerkmale:")
print(
    feature_opt_df.nlargest(20, "gewicht")
    .to_string(index=False)
)


# Klassenscores der optimierten logistischen Regression
wahrscheinlichkeiten = lr_optimiert.predict_proba(X_test_opt)[:, 1]

# Ursprüngliche Ausschreibungsdaten zuordnen
ergebnisse = df.loc[X_test.index].copy()

ergebnisse["vorhersage"] = y_pred_opt
ergebnisse["wahrscheinlichkeit_it"] = wahrscheinlichkeiten

# Klassifikationsergebnisse kategorisieren
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

ergebnisse["klassifikationstyp"] = ergebnisse.apply(
    klassifikationstyp,
    axis=1
)

# Interessante Klassifikationen untersuchen
for typ in [
    "True Positive",
    "False Positive",
    "False Negative"
]:
    auswahl = ergebnisse[
        ergebnisse["klassifikationstyp"] == typ
    ].sort_values(
        "wahrscheinlichkeit_it",
        ascending=False
    )

    print(f"\n{typ}: {len(auswahl)}")

    for _, zeile in auswahl.iterrows():
        print("\n----------------------------")
        print("Notice-ID:", zeile["notice_id"])
        print("CPV-Codes:", zeile["cpv_codes"])
        print(
            "IT-Score:",
            round(zeile["wahrscheinlichkeit_it"], 4)
        )
        print("Text:", zeile["text"][:1200])


from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# Unterschiedliche Entscheidungsschwellen testen
thresholds = [
    0.50, 0.45, 0.40, 0.35,
    0.30, 0.25, 0.20, 0.15, 0.10
]

threshold_ergebnisse = []

for threshold in thresholds:

    y_pred_threshold = (
        wahrscheinlichkeiten >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred_threshold,
        labels=[0, 1]
    ).ravel()

    threshold_ergebnisse.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(
            y_test, y_pred_threshold
        ),
        "Precision": precision_score(
            y_test, y_pred_threshold,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred_threshold
        ),
        "F1": f1_score(
            y_test, y_pred_threshold
        ),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Kandidaten": tp + fp
    })

threshold_df = pd.DataFrame(threshold_ergebnisse)

print("\nThreshold-Analyse:")
print(
    threshold_df.round(3)
    .to_string(index=False)
)

