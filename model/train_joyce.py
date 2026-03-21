import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from build_model import WineModel

# Charger le dataset
df = pd.read_csv('data/dataset.csv')

# Nettoyer (garder quelques colonnes utiles)
df = df[['country', 'price', 'points', 'variety']]
df.dropna(inplace=True)

# Créer la cible : bon vin = 1 si points >= 90, sinon 0
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

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir en tenseurs
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Créer le modèle
model = WineModel(X_train.shape[1])

# Définir loss et optimiseur
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entraînement
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Sauvegarder le modèle
torch.save(model.state_dict(), "model_joyce.pth")