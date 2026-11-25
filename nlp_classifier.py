
import re
import joblib
import pandas as pd

from pathlib import Path


# ==========================================
# 1. Konfiguration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODELLDATEI = (
    BASE_DIR / "ausschreibungsradar_nlp_modell.joblib"
)

CPV_MUSTER = r"\b\d{8}(?:-\d)?\b"


# ==========================================
# 2. Text-Preprocessing
# ==========================================

def bereinige_text(text):

    if not isinstance(text, str):
        return ""

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

    # CPV-Codes entfernen
    text = re.sub(
        CPV_MUSTER,
        " ",
        text
    )

    # Bekanntmachungs-IDs entfernen
    text = re.sub(
        r"Bekanntmachungs-ID:\s*\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Technische Identifikatoren entfernen
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

    # Leerzeichen normalisieren
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# 3. Wiederverwendbare NLP-Komponente
# ==========================================

class NLPClassifier:

    def __init__(self, modellpfad=MODELLDATEI):

        modellpfad = Path(modellpfad)

        if not modellpfad.exists():
            raise FileNotFoundError(
                f"Modelldatei nicht gefunden: {modellpfad}"
            )

        modellpaket = joblib.load(modellpfad)

        self.pipeline = modellpaket["pipeline"]
        self.modellname = modellpaket["modellname"]
        self.threshold = float(modellpaket["threshold"])
        self.trainingsdokumente = modellpaket[
            "trainingsdokumente"
        ]

    # ======================================
    # Einzelne Bekanntmachung klassifizieren
    # ======================================

    def klassifiziere(self, text):

        text_clean = bereinige_text(text)

        if not text_clean:
            return {
                "text_clean": "",
                "it_score": None,
                "it_relevant": None,
                "status": "Kein auswertbarer Text"
            }

        score = float(
            self.pipeline.predict_proba(
                [text_clean]
            )[0, 1]
        )

        relevant = score >= self.threshold

        return {
            "text_clean": text_clean,
            "it_score": score,
            "it_relevant": relevant,
            "status": "Klassifiziert"
        }

    # ======================================
    # Mehrere Bekanntmachungen klassifizieren
    # ======================================

    def klassifiziere_dataframe(
        self,
        df,
        textspalte="text"
    ):

        if textspalte not in df.columns:
            raise ValueError(
                f"Textspalte fehlt: {textspalte}"
            )

        ergebnis = df.copy()

        ergebnis["text_clean"] = (
            ergebnis[textspalte]
            .apply(bereinige_text)
        )

        gueltig = ergebnis["text_clean"].ne("")

        ergebnis["it_score"] = float("nan")

        ergebnis["it_relevant"] = pd.Series(
            pd.NA,
            index=ergebnis.index,
            dtype="boolean"
        )

        ergebnis["status"] = (
            "Kein auswertbarer Text"
        )

        if gueltig.any():

            scores = self.pipeline.predict_proba(
                ergebnis.loc[gueltig, "text_clean"]
            )[:, 1]

            ergebnis.loc[
                gueltig, "it_score"
            ] = scores

            ergebnis.loc[
                gueltig, "it_relevant"
            ] = scores >= self.threshold

            ergebnis.loc[
                gueltig, "status"
            ] = "Klassifiziert"

        return ergebnis

    # ======================================
    # Relevante Bekanntmachungen auswählen
    # ======================================

    def filtere_relevante(self, df):

        if "it_relevant" not in df.columns:
            raise ValueError(
                "DataFrame wurde noch nicht klassifiziert."
            )

        relevante = df[
            df["it_relevant"].fillna(False)
        ].copy()

        return relevante.sort_values(
            "it_score",
            ascending=False
        ).reset_index(drop=True)

    # ======================================
    # Modellinformationen
    # ======================================

    def modellinformationen(self):

        return {
            "modell": self.modellname,
            "threshold": self.threshold,
            "trainingsdokumente": (
                self.trainingsdokumente
            )
        }


# ==========================================
# 4. Funktionstest
# ==========================================

if __name__ == "__main__":

    classifier = NLPClassifier()

    print("\nModellinformationen:")

    for name, wert in (
        classifier.modellinformationen().items()
    ):
        print(f"{name}: {wert}")

    beispiele = pd.DataFrame([
        {
            "id": 1,
            "text": (
                "Beschaffung von Microsoft 365 Lizenzen "
                "einschließlich Migration, Installation "
                "und technischem Support."
            )
        },
        {
            "id": 2,
            "text": (
                "Sanierung eines Schulgebäudes "
                "einschließlich Dacharbeiten, "
                "Maurerarbeiten und Fassadendämmung."
            )
        },
        {
            "id": 3,
            "text": (
                "Einführung einer KI-gestützten "
                "Softwarelösung zur automatisierten "
                "Verarbeitung von Dokumenten."
            )
        },
        {
            "id": 4,
            "text": (
                "Lieferung von Büromöbeln, "
                "Schreibtischen und ergonomischen "
                "Bürostühlen."
            )
        },
        {
            "id": 5,
            "text": (
                "Beschaffung und Implementierung "
                "einer Firewall-Lösung einschließlich "
                "Netzwerksicherheit und Wartung."
            )
        },
        {
            "id": 6,
            "text": (
                "Installation von Cat.7-Datenleitungen "
                "und strukturierter Netzwerkverkabelung "
                "im Rahmen einer Schulsanierung."
            )
        },
        {
            "id": 7,
            "text": ""
        }
    ])

    ergebnisse = (
        classifier.klassifiziere_dataframe(beispiele)
    )

    print("\n" + "=" * 70)
    print("KLASSIFIKATIONSERGEBNISSE")
    print("=" * 70)

    print(
        ergebnisse[
            ["id", "it_score", "it_relevant", "status"]
        ].round(4).to_string(index=False)
    )

    # Einzel- und Batch-Klassifikation vergleichen
    for _, zeile in ergebnisse.iterrows():

        einzelergebnis = classifier.klassifiziere(
            zeile["text"]
        )

        if pd.notna(zeile["it_score"]):

            assert abs(
                einzelergebnis["it_score"]
                - zeile["it_score"]
            ) < 1e-10

            assert (
                einzelergebnis["it_relevant"]
                == zeile["it_relevant"]
            )

        else:

            assert einzelergebnis["it_score"] is None
            assert einzelergebnis["it_relevant"] is None

    print("\nKonsistenzprüfung bestanden.")

    relevante = classifier.filtere_relevante(
        ergebnisse
    )

    print("\n" + "=" * 70)
    print("VORAUSGEWÄHLTE BEKANNTMACHUNGEN")
    print("=" * 70)

    for _, zeile in relevante.iterrows():

        print(f"\nID: {zeile['id']}")
        print(f"IT-Score: {zeile['it_score']:.4f}")
        print(f"Text: {zeile['text']}")

    print("\nZusammenfassung:")

    print("Eingabedokumente:", len(beispiele))

    print(
        "Klassifizierte Dokumente:",
        ergebnisse["it_score"].notna().sum()
    )

    print(
        "Vorausgewählte Dokumente:",
        len(relevante)
    )

    print(
        "Nicht auswertbare Dokumente:",
        ergebnisse["it_score"].isna().sum()
    )

    print("\nNLP-Schnittstelle erfolgreich getestet.")
