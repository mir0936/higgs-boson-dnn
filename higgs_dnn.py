import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("Loading CERN ATLAS collision data...")
df = pd.read_csv('training.csv')

# Note: Took me hours to figure out why my loss was exploding with NaNs. CERN uses -999.0 for missing sensor readings instead of standard nulls. Replacing with 0.0 fixes the tensor math.
df = df.replace(-999.0, 0.0)

X = df.drop(['EventId', 'Weight', 'Label'], axis=1).values
y = np.where(df['Label'] == 's', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class HiggsDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = HiggsDNN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
batch_size = 2048
print("Training Deep Neural Network...")

losses = []
for epoch in range(epochs):
    permutation = torch.randperm(X_train.size()[0])
    epoch_loss = 0.0
    
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / (X_train.size()[0] / batch_size)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

with torch.no_grad():
    test_preds = model(X_test)
    test_preds = (test_preds >= 0.5).float()
    accuracy = (test_preds == y_test).float().mean()
    print(f"\nFinal Test Accuracy: {accuracy.item()*100:.2f}%")

# Basic Training Loss Curve
plt.figure()
plt.plot(range(1, epochs+1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DNN Training Loss')
plt.legend()
plt.savefig('higgs_training.png')
print("Saved basic training curve to higgs_training.png.")

# Basic ROC Curve
from sklearn.metrics import roc_curve, auc
with torch.no_grad():
    y_score = model(X_test).numpy()
    
fpr, tpr, _ = roc_curve(y_test.numpy(), y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (Background Accepted)')
plt.ylabel('True Positive Rate (Signal Accepted)')
plt.title('Higgs Boson Classifier ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('higgs_roc.png')
print("Saved evaluation curve to higgs_roc.png.")
