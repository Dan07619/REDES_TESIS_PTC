# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 03:09:47 2023

@author: nama_
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
    
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load data from csv file
df = pd.read_csv('C:\\Users\\nama_\\DownLoads\\wetransfer_tesis_2023-07-25_0023\\TESIS\\dataset_final_sorted_2.4.3_CSV_prueba_red.csv')

# Split data into training and testing sets

X = df[['ODD_at_HOMO_Max', 'ODD_at_LUMO_Max','Tc(K)',
        'Dipole Moment',  'Theta_Inverse_Max', 
        'Intensive_Entropy_Max', 'Theta_Inverse_Min', 'thermophilicity_intensive_MAX',
        'Intensive_Atomic_softness_Max', 'molecular volume']].values

y = df['Pc(bar)'].values
# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=34)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=34)

# Scale the data to have zero mean and unit variance for input features (X)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_val = sc_X.transform(X_val)
X_test = sc_X.transform(X_test)

# Scale the data to have zero mean and unit variance for the target variable (y)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
y_val = sc_y.transform(y_val.reshape(-1, 1))
y_test = sc_y.transform(y_test.reshape(-1, 1))

# Convert data to PyTorch tensors
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_val = torch.Tensor(X_val)
y_val = torch.Tensor(y_val)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)


# Create DataLoader objects for training, validation, and test data
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)  # Create a DataLoader object for validation data
test_data = TensorDataset(X_test, y_test)

batch_size = 64*2

# Create DataLoader objects for training, validation, and test data
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, )
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, )

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 15)
        self.fc2 = nn.Linear(15, 5)
        self.fc3 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# Training loop
train_losses = []
val_losses = []
test_losses = []

for epoch in range(1500):
    net.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = net(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    


# Validation loop
    with torch.no_grad():
        net.eval()
        val_loss = 0.0
        for X_val_batch, y_val_batch in val_loader:
            outputs_val = net(X_val_batch)
            loss_val = criterion(outputs_val, y_val_batch)
            val_loss += loss_val.item() * X_val_batch.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

# Test loop
    with torch.no_grad():
        net.eval()
        test_loss = 0.0
        for X_test_batch, y_test_batch in test_loader:
            outputs_test = net(X_test_batch)
            loss_test = criterion(outputs_test, y_test_batch)
            test_loss += loss_test.item() * X_test_batch.size(0)

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)


    if epoch % 1 == 0:
        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, 1000, train_loss, val_loss, test_loss))




# Plot the losses for the training set, validation set, and test set
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss',  )
plt.plot(val_losses, label='Validation Loss',linestyle = "--")
plt.plot(test_losses, label='Test Loss',  linestyle = "--")  # Add this line to plot the test loss

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation, and Test Losses Pc')
plt.legend()
plt.show()



# Obtain predictions for each output separately
predicted = outputs_test
    

# set the model to evaluation mode
net.eval()
with torch.no_grad():
    predicted = net(X_test)
    test_loss = criterion(predicted, y_test)

# create empty arrays to store predicted and actual values
predicted = np.array(predicted)

actual2 = np.array(y_test)


# loop through the test set and generate predictions
with torch.no_grad():
    for x, y in test_loader:
        x = x.to('cpu')
        y = y.to('cpu')
        y_pred = net(x)
        predicted = np.append(predicted, y_pred.cpu().detach().numpy())

        actual2 = np.append(actual2, y.cpu().detach().numpy())


# calculate the mean squared error for each output
mse2 = np.mean(np.square(predicted - actual2))

print("Mean Squared Error for output 2: {:.4f}".format(mse2))



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Deshacer la transformación de escalado en las etiquetas y_test
actual_unscaled = sc_y.inverse_transform(actual2.reshape(-1, 1))

# Deshacer la transformación de escalado en las predicciones (outputs_test)
predicted_unscaled = sc_y.inverse_transform(predicted.reshape(-1, 1))

# plot the predicted and actual values for each output separately
plt.figure(figsize=(15,6))
plt.subplot(2, 1, 1)
plt.plot(actual_unscaled, label="Actual")
plt.plot(predicted_unscaled, label="Predicted")
plt.legend()
plt.title("Predicted vs Actual for Critical Temperature")
plt.xlabel("Sample")
plt.ylabel("Tc(K)")

# plot the residuals for each output
residuals = actual_unscaled - predicted_unscaled

plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.scatter(predicted_unscaled, residuals)
plt.title("Residual Plot for Critical Temperature ")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.axhline(y=0, color='r', linestyle='-')


# calculate the mean squared error for each output
mse2 = np.mean(np.square(predicted - actual2))

print("Mean Squared Error for output 2: {:.4f}".format(mse2))


# plot the predicted and actual values for each output separately
plt.figure(figsize=(15,6))
plt.subplot(2, 1, 1)
plt.plot(actual2, label="Actual")
plt.plot(predicted, label="Predicted")
plt.legend()
plt.title("Predicted vs Actual for Acentric Factor ")
plt.xlabel("Sample")
plt.ylabel("Value")


# plot the residuals for each output
residuals = actual2 - predicted

plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.scatter(predicted, residuals)
plt.title("Residual Plot for Acentric Factor")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.axhline(y=0, color='r', linestyle='-')
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate MSE, RMSE, and MAE
mse = mean_squared_error(actual2, predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual2, predicted)

# Calculate R-squared
r2 = r2_score(actual2, predicted)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")

# Obtener predicciones del modelo para el conjunto de prueba
with torch.no_grad():
    net.eval()
    predictions_pc = net(X_test)

# Convertir las predicciones y las etiquetas a arrays de NumPy
predictions_pc = predictions_pc.cpu().numpy().flatten()
y_test5 = y_test.cpu().numpy().flatten()

# Guardar las variables en un archivo
np.savez('variables_predicciones5.npz', predictions_pc=predictions_pc,y_test5=y_test5)
# Graficar el gráfico de calibración
plt.figure(figsize=(8, 6))
plt.plot(predictions_pc, y_test, 'o', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Idealmente calibrado')
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.title('Gráfico de Calibración')
plt.legend()
plt.show()

