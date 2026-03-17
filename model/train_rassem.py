import pickle
from build_model import build_model_cnn, get_callbacks, plot_history

def train_cnn():
    """Charge les données, entraîne le CNN1D et enregistre le modèle."""
    with open('processed_data.pkl', 'rb') as f:
         X_train, X_test, y_train, y_test, max_words, max_len = pickle.load(f)

    print("Construction du CNN 1D...")
    model = build_model_cnn(max_words, max_len)
    model.summary()
    
    print("Début de l'entraînement du CNN 1D...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        callbacks=get_callbacks()
    )
    
    model.save('cnn_wine_model.keras')
    print("Modèle CNN sauvegardé sous cnn_wine_model.keras")
    plot_history(history, model_name="CNN_1D")

if __name__ == "__main__":
    train_cnn()
