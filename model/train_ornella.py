import pickle
from build_model import build_model_lstm, get_callbacks, plot_history

def train_lstm():
    """Charge les données, entraîne le BiLSTM et enregistre le modèle."""
    with open('processed_data.pkl', 'rb') as f:
         X_train, X_test, y_train, y_test, max_words, max_len = pickle.load(f)

    print("Construction du BiLSTM...")
    model = build_model_lstm(max_words, max_len)
    model.summary()
    
    print("Début de l'entraînement du BiLSTM...")
    # Le LSTM est plus lourd, on utilise un batch_size légèrement plus grand
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.2,
        callbacks=get_callbacks()
    )
    
    model.save('lstm_wine_model.keras')
    print("Modèle LSTM sauvegardé sous lstm_wine_model.keras")
    plot_history(history, model_name="BiLSTM")

if __name__ == "__main__":
    train_lstm()
