import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from build_model import WineModel

# Charger le dataset
df = pd.read_csv('../data/dataset.csv')

# Ornella ajoute plus de features
df = df[['country', 'price', 'points', 'variety', 'province']]
df.dropna(inplace=True)

# Créer la cible
df['target'] = (df['points'] >= 90).astype(int)

# Encoder country, variety, province
le_country = LabelEncoder()
le_variety = LabelEncoder()
le_province = LabelEncoder()

df['country'] = le_country.fit_transform(df['country'])
df['variety'] = le_variety.fit_transform(df['variety'])
df['province'] = le_province.fit_transform(df['province'])

# Features et cible
X = df[['country', 'price', 'variety', 'province']].values
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

model = WineModel(X_train.shape[1])

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

torch.save(model.state_dict(), "model_ornella.pth")
