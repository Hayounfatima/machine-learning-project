import pandas as pd

def run(input_csv="HR_Analytics.csv", output_csv="df_cleaned.csv"):
    """
    Étape A : Analyse exploratoire + traitement des valeurs manquantes
    """

    # 1) Charger le dataset ORIGINAL

    df = pd.read_csv(input_csv)

    print(" Informations générales ")
    print("Shape (lignes, colonnes) :", df.shape)
    print("\nTypes des colonnes :")
    print(df.info())

    # 2) Vérifier les valeurs manquantes AVANT traitement
    print("\n=== Valeurs manquantes AVANT nettoyage ===")
    missing_before = df.isnull().sum()
    print(missing_before)

    # 3) Traitement des valeurs manquantes

    if "YearsWithCurrManager" in df.columns:
        median_value = df["YearsWithCurrManager"].median()
        df["YearsWithCurrManager"] = df["YearsWithCurrManager"].fillna(median_value)

        print(
            f"\nYearsWithCurrManager : NaN remplacés par la médiane = {median_value}"
        )

    # 4) Vérifier les valeurs manquantes APRÈS traitement
    print("\n Valeurs manquantes APRÈS nettoyage ")
    missing_after = df.isnull().sum()
    print(missing_after)

    # 5) Distribution de la variable cible
    print("\n=== Distribution de Attrition (%) ===")
    print(df["Attrition"].value_counts(normalize=True) * 100)

    # 6) Sauvegarde du dataset nettoyé

    df.to_csv(output_csv, index=False)

    print(f"\n Dataset nettoyé sauvegardé dans : {output_csv}")
    print("Le fichier original n'a PAS été modifié.")

# 7) Point d'entrée du script
if __name__ == "__main__":
    run()
