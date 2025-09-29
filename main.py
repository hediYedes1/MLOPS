# main.py
import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn Prediction")
    parser.add_argument("--action", type=str, required=True,
                        choices=["prepare", "train", "evaluate", "predict", "all"],
                        help="Étape à exécuter")
    parser.add_argument("--model", type=str, default="classifier.joblib",
                        help="Nom du fichier modèle")
    args = parser.parse_args()

    if args.action == "prepare":
        X_train, X_test, y_train, y_test = prepare_data()
        print("✅ Données préparées.")
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    elif args.action == "train":
        X_train, X_test, y_train, y_test = prepare_data()
        model = train_model(X_train, y_train)
        save_model(model, args.model)
        print("✅ Modèle entraîné et sauvegardé.")

    elif args.action == "evaluate":
        X_train, X_test, y_train, y_test = prepare_data()
        model = load_model(args.model)
        evaluate_model(model, X_test, y_test)

    elif args.action == "predict":
        model = load_model(args.model)
        sample = [[850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]]
        pred = model.predict(sample)
        print(f"✅ Prédiction pour l’échantillon : {pred}")

    elif args.action == "all":
        # Préparer
        X_train, X_test, y_train, y_test = prepare_data()
        print("✅ Données préparées.")

        # Entraîner
        model = train_model(X_train, y_train)
        print("✅ Modèle entraîné.")

        # Évaluer
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print("\n📊 Résultats du modèle :")
        print(f"Accuracy   : {acc:.4f}")
        print(f"Précision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1-score   : {f1:.4f}")
        print(f"MSE        : {mse:.4f}")

        # Sauvegarder + recharger
        save_model(model, args.model)
        loaded_model = load_model(args.model)

        # Prédiction exemple
        sample = [[850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]]
        pred = loaded_model.predict(sample)
        print(f"\n✅ Prédiction pour l’échantillon : {pred}")


if __name__ == "__main__":
    main()
