
import pandas as pd

# Gespeicherte Ausschreibungen laden
df = pd.read_csv("ausschreibungen_2023_1000.csv")

print("Anzahl der Datensätze:", len(df))

print("\nFehlende Werte:")
print(df.isnull().sum())

print("\nDoppelte Kennungen:")
print(df.duplicated(subset=["identifier"]).sum())

print("\nIdentische Ausschreibungstexte:")
print(df.duplicated(subset=["text"]).sum())

print("\nVerteilung der IT-Kandidaten:")
print(df["it_kandidat"].value_counts())


print("\nDie 10 häufigsten Kennungen:")
print(df["identifier"].value_counts().head(10))

print("\nBeispiel für eine wiederholte Kennung:")

haeufigste_id = df["identifier"].value_counts().idxmax()

beispiele = df[df["identifier"] == haeufigste_id]

for _, zeile in beispiele.head(5).iterrows():
    print("\n----------------------------")
    print("Identifier:", zeile["identifier"])
    print("Dateiname:", zeile["filename"])
    print("Datum:", zeile["date"])
    print("URL:", zeile["url"])
    print("Text:", zeile["text"][:300])


print("\nPrüfung der Dateinamen:")
print("Doppelte Dateinamen:", df["filename"].duplicated().sum())
print("Eindeutige Dateinamen:", df["filename"].nunique())

# Individuelle TED-Bekanntmachungsnummer extrahieren
df["notice_id"] = df["filename"].str.extract(
    r"(\d+-\d{4})\.xml$"
)

print("\nNeue Ausschreibungs-IDs:")
print(df[["filename", "notice_id"]].head(10))

print("\nFehlende Ausschreibungs-IDs:")
print(df["notice_id"].isnull().sum())

print("\nDoppelte Ausschreibungs-IDs:")
print(df["notice_id"].duplicated().sum())
