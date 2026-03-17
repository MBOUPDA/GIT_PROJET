import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from build_model import WineModel

# Charger le dataset
df = pd.read_csv('../data/dataset.csv')

# Nettoyage : garder quelques colonnes utiles
df = df[['country', 'price', 'points', 'variety']]
df.dropna(inplace=True)

# Cible : bon vin si points >= 90
df['target'] = (df['points'] >= 90).astype(int)

# Encoder les variables catégorielles
le_country = LabelEncoder()
le_variety = LabelEncoder()

df['country'] = le_country.fit_transform(df['country'])
df['variety'] = le_variety.fit_transform(df['variety'])

# Features (X) et cible (y)
X = df[['country', 'price', 'variety']].values
y = df['target'].values

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Créer le modèle avec WineModel
model = WineModel(X_train.shape[1])

# Architecture optimisée par Rassem
model.net = torch.nn.Sequential(
    torch.nn.Linear(X_train.shape[1], 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.
