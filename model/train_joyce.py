import pickle
from build_model import build_model_dnn, get_callbacks, plot_history

def train_dnn():
    """Charge les données, entraîne le DNN et enregistre le modèle."""
    # Chargement des données prétraitées
    with open('processed_data.pkl', 'rb') as f:
         X_train, X_test, y_train, y_test, max_words, max_len = pickle.load(f)

    # Construction
    print("Construction du DNN...")
    model = build_model_dnn(max_words, max_len)
    model.summary()
    
    # Entraînement avec EarlyStopping
    print("Début de l'entraînement du DNN...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        callbacks=get_callbacks()
    )
    
    # Enregistrement au format TensorFlow/Keras standard (recommandé plutôt que pkl pour les NNs)
    model.save('dnn_wine_model.keras')
    print("Modèle DNN sauvegardé sous dnn_wine_model.keras")
    
    # Plot et évaluation
    plot_history(history, model_name="DNN_Base")

if __name__ == "__main__":
    train_dnn()
