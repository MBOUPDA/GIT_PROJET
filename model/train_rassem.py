import pandas as pd
import torch
import copy 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from build import WineModel


# Nettoyage
df = pd.read_csv('dataset.csv')
df = df[['country', 'price', 'points', 'variety']]
df.dropna(inplace=True)
df['target'] = (df['points'] >= 90).astype(int)


# Encoder
le_country = LabelEncoder()
le_variety = LabelEncoder()

df['country'] = le_country.fit_transform(df['country'])
df['variety'] = le_variety.fit_transform(df['variety'])

X = df[['country', 'price', 'variety']].values
y = df['target'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Architecture optimisée
model = WineModel(X_train.shape[1])
model.net = torch.nn.Sequential(
    torch.nn.Linear(X_train.shape[1], 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
    torch.nn.Sigmoid()
)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#Entrainnement
epochs = 1000 
patience = 15 
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_weights = copy.deepcopy(model.state_dict())

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

def calculate_accuracy(y_pred, y_true):
    predictions = (y_pred >= 0.5).float() 
    correct = (predictions == y_true).sum().item()
    return correct / len(y_true)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_acc = calculate_accuracy(outputs, y_train)
    model.eval() 
    with torch.no_grad(): 
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        val_acc = calculate_accuracy(val_outputs, y_test)
        
    #Sauvegarde des métriques
    history['train_loss'].append(loss.item())
    history['val_loss'].append(val_loss.item())
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    if epoch % 10 == 0: print(f"Epoch {epoch:03d} | accuracy : {train_acc:.4f} - val_accuracy : {val_acc:.4f} - loss : {loss.item():.4f} - val_loss : {val_loss.item():.4f}")
        
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        epochs_no_improve = 0
        best_model_weights = copy.deepcopy(model.state_dict()) 
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly Stopping déclenché à l'époque {epoch}!")
            break


#Sauvegarde
model.load_state_dict(best_model_weights)
torch.save(model.state_dict(), "model_rassem_best.pth")
print("Modèle optimal sauvegardé sous 'model_rassem_best.pth'.")


#Visualisation
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train', color='blue', linewidth=2)
plt.plot(history['val_loss'], label='Validation', color='red', linestyle='--', linewidth=2)
plt.title('Fonction de cout')
plt.xlabel('Époques')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train', color='blue', linewidth=2)
plt.plot(history['val_acc'], label='Validation', color='red', linestyle='--', linewidth=2)
plt.title('Précision')
plt.xlabel('Époques')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
