import argparse

import missingvalues
import preprocessing_smote 
import SMOTE_from_scratch
import visualisation

#(comparaison sklearn)
try:
    import sklearn_comparaison
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - HR Attrition")
    parser.add_argument("--step", type=str, default="all",
                        choices=["A", "B", "C", "D", "E", "SK", "all"],
                        help="Choisir une étape à exécuter ou all.")
    args = parser.parse_args()

    if args.step in ["A", "all"]:
        print("\n=== Étape A: EDA + Cleaning ===")
        missingvalues.run()

    if args.step in ["B", "all"]:
        print("\n=== Étape B: Split + Standardisation ===")
        preprocessing_smote .run()

    if args.step in ["C", "all"]:
        print("\n=== Étape C: SMOTE from scratch ===")
        SMOTE_from_scratch.run()

  

    if args.step in ["E", "all"]:
        print("\n=== Étape E: Évaluation + Visualisation ===")
        visualisation.run()

    if args.step in ["SK", "all"]:
        if not HAS_SKLEARN:
            print("\n[WARN] sklearn_comparaison.py introuvable ou sklearn non installé.")
        else:
            print("\n=== Comparaison sklearn ===")
            sklearn_comparaison.run()


if __name__ == "__main__":
    main()
