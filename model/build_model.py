import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. PREPROCESSING ---
def load_and_preprocess_data(filepath, max_words=20000, max_len=100):
    """Charge, nettoie et prépare les données pour l'entraînement."""
    print("Chargement des données...")
    df = pd.read_csv(filepath)
    
    # Nettoyage : suppression des lignes sans description ou sans points
    df = df.dropna(subset=['description', 'points'])
    
    # Transformation de la Target (Classification binaire >= 90 points)
    df['is_excellent'] = (df['points'] >= 90).astype(int)
    
    # Text Processing (Tokenization et Padding)
    texts = df['description'].astype(str).tolist()
    labels = df['is_excellent'].values
    
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
    
    # Sauvegarde du tokenizer en .pkl
    with open('tokenizer_vins.pkl', 'wb') as f:
         pickle.dump(tokenizer, f)
    print("Tokenizer sauvegardé en tokenizer_vins.pkl")
    
    return X_train, X_test, y_train, y_test, tokenizer, max_words, max_len

# --- 2. BUILD MODELS ---
# Modèle 1 : Réseau de Neurones Profond (DNN basique)
def build_model_dnn(max_words, max_len):
    """Construit un modèle de neurones profond sur les embeddings de mots."""
    inputs = tf.keras.Input(shape=(max_len,))
    x = layers.Embedding(input_dim=max_words, output_dim=64, input_length=max_len)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Modèle 2 : LSTM (Long Short-Term Memory)
def build_model_lstm(max_words, max_len):
    """Construit un modèle récurrent adapté aux séquences de texte complexes (critiques de vin)."""
    inputs = tf.keras.Input(shape=(max_len,))
    x = layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_len)(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Modèle 3 : CNN 1D pour le Traitement du Langage Naturel
def build_model_cnn(max_words, max_len):
    """Construit un réseau convolutif 1D, très rapide et excellent pour extraire les mots-clés clés d'une description."""
    inputs = tf.keras.Input(shape=(max_len,))
    x = layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_len)(inputs)
    x = layers.Conv1D(128, 5, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 3. CALLBACKS & PLOTS (Outils utilitaires) ---
def get_callbacks():
    """Génère l'EarlyStopping et le ModelCheckpoint."""
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    return [early_stop]

def plot_history(history, model_name="Modèle"):
    """Génère les graphiques pour loss, val_loss, accuracy, val_accuracy."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Accuracy - {model_name}')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss - {model_name}')
    plt.legend()
    
    plt.savefig(f'{model_name}_learning_curves.png')
    plt.show()

# --- Exécution pour charger les données (utile pour les 3 prochains scripts) ---
if __name__ == '__main__':
    # Modifie le chemin ci-dessous vers ton fichier local
    filepath = "winemag-data-130k-v2.csv"
    X_train, X_test, y_train, y_test, tokenizer, max_words, max_len = load_and_preprocess_data(filepath)
    
    # Sauvegarde des données pré-traitées pour les utiliser dans les autres scripts
    with open('processed_data.pkl', 'wb') as f:
         pickle.dump((X_train, X_test, y_train, y_test, max_words, max_len), f)
    print("Données prétraitées sauvegardées en processed_data.pkl")
