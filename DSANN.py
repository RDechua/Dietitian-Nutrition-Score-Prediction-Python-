#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:15:44 2025

@author: rubenodehcua
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sklearn as sk

#%%
fname = '/Users/rubenodehcua/Desktop/DSA/testdata.csv' # Uses only columns from reg_col


df_train = pd.read_csv(fname)

data = df_train.values
data[:, :-1][np.isnan(data[:, :-1])] = 0 # changes all columns except last from nan value to 0

#%%

nan_entries = data[np.isnan(data[:, 11])]
non_nan_entries = data[~np.isnan(data[:, 11])]

X_nan = nan_entries[:, :-1]

X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(non_nan_entries[:, :-1], non_nan_entries[:, -1], test_size=0.2, random_state=42)

#%%
#

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        xi = torch.tensor(self.X[i], dtype=torch.float32)
        if self.y is not None:
            yi = torch.tensor(self.y[i], dtype=torch.float32)
            return xi, yi
        return xi  # test set (no labels)

class RegressionNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(X_train.shape[1], 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)  # Output is a single value
        )

    def forward(self, x):
        return self.model(x)

#%%
#

train_dataset = SimpleDataset(X_train, y_train)
val_dataset = SimpleDataset(X_val, y_val)
nan_dataset = SimpleDataset(X_nan)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
nan_loader = torch.utils.data.DataLoader(nan_dataset, batch_size=32)

#%%

model = RegressionNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

num_epochs = 50
train_losses = []
val_losses = []
avg_differences = []

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    total_train_samples = 0
    
    for xb, yb in train_loader:
        pred = model(xb).squeeze()
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * len(xb)
        total_train_samples += len(xb)
    
    epoch_train_loss = total_train_loss / total_train_samples
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    total_abs_diff = 0
    total_val_samples = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            total_val_loss += loss.item() * len(xb)

            abs_diff = torch.abs(pred - yb).sum().item()
            total_abs_diff += abs_diff
            total_val_samples += len(xb)

    epoch_val_loss = total_val_loss / total_val_samples
    avg_diff = total_abs_diff / total_val_samples

    val_losses.append(epoch_val_loss)
    avg_differences.append(avg_diff)

    print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Avg Diff: {avg_diff:.4f}")
    
#%%

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss (MSE)')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss (MSE)')
plt.title('Training and Validation Loss (MSE) vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), avg_differences)
plt.title('Average Absolute Difference vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Avg |Prediction - Actual|')
plt.grid(True)
plt.show()

#%%

model.eval()
nan_preds = []

with torch.no_grad():
    for xb in nan_loader:
        pred = model(xb).squeeze()
        nan_preds.extend(pred.cpu().numpy())

#%%

model.eval()  # Ensure the model is in evaluation mode
y_train_preds = []
y_train_actual = []

with torch.no_grad():
    for xb, yb in train_loader:
        preds = model(xb).squeeze()
        y_train_preds.extend(preds.cpu().numpy())
        y_train_actual.extend(yb.cpu().numpy())

y_train_preds = np.array(y_train_preds)
y_train_actual = np.array(y_train_actual)
