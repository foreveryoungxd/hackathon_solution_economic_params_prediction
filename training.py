import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import r2_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_slice = np.load('часть_9.npy')
target_slice = np.load('часть12_9.npy')

data_slice = data_slice[:50001, :, :]
target_slice = target_slice[:50001, :, :]

print(data_slice.shape)
print(target_slice.shape)

X_train = torch.tensor(data_slice, dtype=torch.float32, requires_grad=True).to(device)
y_train = torch.tensor(target_slice, dtype=torch.float32, requires_grad=True).to(device)

batch_size = 64
input_size = 600  # Размерность входных данных (200x3 = 600)
hidden_size = 256  # Размер скрытого слоя LSTM
output_size = 15  # Размерность выходных данных (15 параметров)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'input': self.X[idx], 'target': self.y[idx]}
        return sample

dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class MeanLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_prob=0):
        super(MeanLSTMModel, self).__init__()
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x_flat = self.flatten(x)
        lstm_out, _ = self.lstm(x_flat.view(x_flat.size(0), -1, input_size))
        output = lstm_out[:, -1, :]
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        output = output.view(output.size(0), -1, 1)
        return output

class QuantileLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_prob=0):
        super(QuantileLSTMModel, self).__init__()
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x_flat = self.flatten(x)
        lstm_out, _ = self.lstm(x_flat.view(x_flat.size(0), -1, input_size))
        output = lstm_out[:, -1, :]
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        output = output.view(output.size(0), -1, 1)
        return output

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        loss = (torch.sum((targets - outputs) ** 2))
        return loss

class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, outputs, targets):
        errors = targets - outputs
        indicator = (errors > 0).float()
        indicator_2 = (errors <= 0).float()
        quantile_loss = ((self.quantile / 100) * indicator + ((1 - (self.quantile / 100)) * indicator_2)) * torch.abs(errors)
        return torch.sum(quantile_loss)


mean_model = MeanLSTMModel(input_size, hidden_size, output_size, dropout_prob=0).to(device)
mean_criterion = nn.MSELoss()
mean_optimizer = torch.optim.Adam(mean_model.parameters(), lr=0.001)

quantile_model_10 = QuantileLSTMModel(input_size, hidden_size, output_size, dropout_prob=0).to(device)
quantile_criterion_10 = QuantileLoss(quantile=10)
quantile_optimizer_10 = torch.optim.Adam(quantile_model_10.parameters(), lr=0.001)

quantile_model_25 = QuantileLSTMModel(input_size, hidden_size, output_size, dropout_prob=0).to(device)
quantile_criterion_25 = QuantileLoss(quantile=25)
quantile_optimizer_25 = torch.optim.Adam(quantile_model_25.parameters(), lr=0.001)

quantile_model_50 = QuantileLSTMModel(input_size, hidden_size, output_size, dropout_prob=0).to(device)
quantile_criterion_50 = QuantileLoss(quantile=50)
quantile_optimizer_50 = torch.optim.Adam(quantile_model_50.parameters(), lr=0.001)

quantile_model_75 = QuantileLSTMModel(input_size, hidden_size, output_size, dropout_prob=0).to(device)
quantile_criterion_75 = QuantileLoss(quantile=75)
quantile_optimizer_75 = torch.optim.Adam(quantile_model_75.parameters(), lr=0.001)

quantile_model_90 = QuantileLSTMModel(input_size, hidden_size, output_size, dropout_prob=0).to(device)
quantile_criterion_90 = QuantileLoss(quantile=90)
quantile_optimizer_90 = torch.optim.Adam(quantile_model_90.parameters(), lr=0.001)

scheduler_mean = torch.optim.lr_scheduler.StepLR(mean_optimizer, step_size=5, gamma=0.2)
scheduler_10 = torch.optim.lr_scheduler.StepLR(quantile_optimizer_10, step_size=5, gamma=0.2)
scheduler_25 = torch.optim.lr_scheduler.StepLR(quantile_optimizer_25, step_size=5, gamma=0.2)
scheduler_50 = torch.optim.lr_scheduler.StepLR(quantile_optimizer_50, step_size=5, gamma=0.2)
scheduler_75 = torch.optim.lr_scheduler.StepLR(quantile_optimizer_75, step_size=5, gamma=0.2)
scheduler_90 = torch.optim.lr_scheduler.StepLR(quantile_optimizer_90, step_size=5, gamma=0.2)

num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        inputs, targets = batch['input'].to(device), batch['target'].to(device)

        mean_optimizer.zero_grad()
        quantile_optimizer_10.zero_grad()
        quantile_optimizer_25.zero_grad()
        quantile_optimizer_50.zero_grad()
        quantile_optimizer_75.zero_grad()
        quantile_optimizer_90.zero_grad()

        outputs_mean = mean_model(inputs)
        outputs_quantile_10 = quantile_model_10(inputs)
        outputs_quantile_25 = quantile_model_25(inputs)
        outputs_quantile_50 = quantile_model_50(inputs)
        outputs_quantile_75 = quantile_model_75(inputs)
        outputs_quantile_90 = quantile_model_90(inputs)

        loss_mean = mean_criterion(outputs_mean, targets)
        loss_quantile_10 = quantile_criterion_10(outputs_quantile_10, targets)
        loss_quantile_25 = quantile_criterion_10(outputs_quantile_25, targets)
        loss_quantile_50 = quantile_criterion_10(outputs_quantile_50, targets)
        loss_quantile_75 = quantile_criterion_10(outputs_quantile_75, targets)
        loss_quantile_90 = quantile_criterion_10(outputs_quantile_90, targets)

        loss_mean.backward()
        loss_quantile_10.backward()
        loss_quantile_25.backward()
        loss_quantile_50.backward()
        loss_quantile_75.backward()
        loss_quantile_90.backward()

        mean_optimizer.step()
        quantile_optimizer_10.step()
        quantile_optimizer_25.step()
        quantile_optimizer_50.step()
        quantile_optimizer_75.step()
        quantile_optimizer_90.step()


        outputs_numpy = outputs_quantile_10.cpu().detach().numpy().reshape(-1, 1)
        targets_numpy = targets.cpu().detach().numpy().reshape(-1, 1)


        r_squared = r2_score(targets_numpy, outputs_numpy)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss_quantile_10.item():.4f}, R²: {r_squared:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss_mean: {loss_mean.item():.4f}, Loss_quantile:{loss_quantile_10.item():.4f}')

    scheduler_mean.step()
    scheduler_10.step()
    scheduler_25.step()
    scheduler_50.step()
    scheduler_75.step()
    scheduler_90.step()


torch.save(mean_model.state_dict(), 'mean_test_model_1.pb')
torch.save(quantile_model_10.state_dict(), 'quantile_test_model_10_1.pb')
torch.save(quantile_model_25.state_dict(), 'quantile_test_model_25_1.pb')
torch.save(quantile_model_50.state_dict(), 'quantile_test_model_50_1.pb')
torch.save(quantile_model_75.state_dict(), 'quantile_test_model_75_1.pb')
torch.save(quantile_model_90.state_dict(), 'quantile_test_model_90_1.pb')








