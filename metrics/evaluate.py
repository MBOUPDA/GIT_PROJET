import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from GIT_PROJET.model.build_model import WineModel  # IMPORTER LA CLASSE EXACTE DU MODÈLE

# Charger le dataset
df = pd.read_csv('data/dataset.csv')
df = df[['country', 'price', 'variety', 'points']].dropna()
df['target'] = (df['points'] >= 90).astype(int)

# Encoder les variables catégorielles
le_country = LabelEncoder()
le_variety = LabelEncoder()
df['country'] = le_country.fit_transform(df['country'])
df['variety'] = le_variety.fit_transform(df['variety'])

# Features et cible
X = df[['country', 'price', 'variety']].values
y = df['target'].values

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convertir en tenseurs
X_test = torch.tensor(X, dtype=torch.float32)
y_test = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Créer le modèle EXACTEMENT comme à l'entraînement
model = WineModel(X_test.shape[1])

# Charger les poids
state_dict = torch.load('model_joyce.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()  # passer en mode évaluation

# Prédictions
with torch.no_grad():
    preds = model(X_test)
    preds = (preds > 0.5).int()

# Calcul des métriques
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Blocage si métriques trop faibles
if accuracy < 0.70:
    print("Accuracy trop faible. Commit bloqué.")
    exit(1)
if precision < 0.70:
    print("Précision trop faible. Commit bloqué.")
    exit(1)
if recall < 0.30:
    print("Recall trop faible. Commit bloqué.")
    exit(1)