import pickle
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_models():
    """Évalue et compare tous les modèles sur le Test Set."""
    # 1. Charger les données de test
    print("Chargement des données de test...")
    with open('../model/processed_data.pkl', 'rb') as f:
         X_train, X_test, y_train, y_test, max_words, max_len = pickle.load(f)

    models_to_evaluate = {
        'DNN_Base': '../model/dnn_wine_model.keras',
        'BiLSTM': '../model/lstm_wine_model.keras',
        'CNN_1D': '../model/cnn_wine_model.keras'
    }
    
    results = []

    for name, path in models_to_evaluate.items():
        try:
            print(f"\n================ Évaluation du modèle: {name} ================")
            model = tf.keras.models.load_model(path)
            
            # Évaluation native Keras (loss et accuracy globales)
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            results.append((name, loss, accuracy))
            print(f"{name} - Loss : {loss:.4f} | Accuracy : {accuracy:.4f}")
            
            # Prédictions (Probabilités en Classes 0 ou 1)
            y_pred_probs = model.predict(X_test, verbose=0)
            y_pred_classes = (y_pred_probs >= 0.5).astype(int)
            
            # Rapport de classification approfondi (Precision, Recall, F1-Score)
            print(f"\nRapport détaillé pour {name} :")
            print(classification_report(y_test, y_pred_classes, target_names=["<90 Points", ">=90 Points"]))
            
        except OSError:
            print(f"Modèle {name} introuvable à l'adresse {path}. As-tu lancé son script d'entraînement ?")

    # Affichage du gagnant
    if results:
        print("\n================ RÉSUMÉ DES RÉSULTATS ================")
        results.sort(key=lambda x: x[2], reverse=True) # Trier par accuracy décroissante
        print("Classement (du meilleur au moins performant) :")
        for i, (name, loss, acc) in enumerate(results):
            print(f"{i+1}. {name} (Accuracy: {acc*100:.2f}%)")

if __name__ == "__main__":
    evaluate_models()
