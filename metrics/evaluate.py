import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

# Charger le modèle
model = torch.load('model_joyce.pth')
model.eval()

# Charger le dataset de test (ou des prédictions)
df = pd.read_csv('../data/dataset.csv')

# Préparer les données de test (pareil que pendant l'entraînement)
df = df[['country', 'price', 'variety', 'points']]
df = df.dropna()

df['target'] = (df['points'] >= 90).astype(int)

# Encoder et normaliser (comme pendant l’entraînement)
df['country'] = df['country'].astype('category').cat.codes
df['variety'] = df['variety'].astype('category').cat.codes

X = df[['country', 'price', 'variety']].values
y = df['target'].values

# Convertir en tenseurs
X_test = torch.tensor(X, dtype=torch.float32)
y_test = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Prédictions
preds = model(X_test)
preds = (preds > 0.5).int()

# Calcul des métriques
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Si les métriques ne sont pas bonnes, on bloque
if accuracy < 0.80:
    print("Accuracy trop faible. Commit bloqué.")
    exit(1)
if precision < 0.75:
    print("Précision trop faible. Commit bloqué.")
    exit(1)
if recall < 0.70:
    print("Recall trop faible. Commit bloqué.")
    exit(1)
