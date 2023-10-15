import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from sklearn.metrics import r2_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



batch_size = 64
new_data_tensor = torch.tensor(np.load('часть_19.npy'), dtype=torch.float32).to(device)


class QuantileLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_prob=0.2):
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

class MeanLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_prob=0.2):
        super(MeanLSTMModel, self).__init__()
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x_flat = self.flatten(x)  # Вытягиваем входные матрицы в вектора
        lstm_out, _ = self.lstm(x_flat.view(x_flat.size(0), -1, input_size))  # Добавляем размерность временных шагов
        output = lstm_out[:, -1, :] # Получаем выход из последнего временного шага
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        output = output.view(output.size(0), -1, 1)
        return output

input_size = 600  # Размерность входных данных (200x3 = 600)
hidden_size = 256  # Размер скрытого слоя LSTM
output_size = 15  # Размерность выходных данных (15 параметров)
class CustomNewDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'input': self.X[idx]}
        return sample

new_dataset = CustomNewDataset(new_data_tensor)
new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

mean_model = MeanLSTMModel(input_size, hidden_size, output_size, num_layers=2, dropout_prob=0).to(device)
quantile_model_10 = QuantileLSTMModel(input_size, hidden_size, output_size, dropout_prob=0).to(device)
quantile_model_25 = QuantileLSTMModel(input_size, hidden_size, output_size, dropout_prob=0).to(device)
quantile_model_50 = QuantileLSTMModel(input_size, hidden_size, output_size, dropout_prob=0).to(device)
quantile_model_75 = QuantileLSTMModel(input_size, hidden_size, output_size, dropout_prob=0).to(device)
quantile_model_90 = QuantileLSTMModel(input_size, hidden_size, output_size, dropout_prob=0).to(device)

mean_model.load_state_dict(torch.load('mean_test_model_1.pb'))
quantile_model_10.load_state_dict(torch.load('quantile_test_model_10_1.pb'))
quantile_model_25.load_state_dict(torch.load('quantile_test_model_25_1.pb'))
quantile_model_50.load_state_dict(torch.load('quantile_test_model_50_1.pb'))
quantile_model_75.load_state_dict(torch.load('quantile_test_model_75_1.pb'))
quantile_model_90.load_state_dict(torch.load('quantile_test_model_90_1.pb'))



predictions_mean = []
predictions_quantile_10 = []
predictions_quantile_25 = []
predictions_quantile_50 = []
predictions_quantile_75 = []
predictions_quantile_90 = []

mean_model.eval()
quantile_model_10.eval()  # Переведем модель в режим оценки (не обучения)
quantile_model_25.eval()
quantile_model_50.eval()
quantile_model_75.eval()
quantile_model_90.eval()

with torch.no_grad():  # Не считаем градиенты, так как это предсказания, не обучение
    for batch_idx, batch in enumerate(new_loader):
        inputs = batch['input'].to(device)
        outputs_mean = mean_model(inputs)
        outputs_10 = quantile_model_10(inputs)
        outputs_25 = quantile_model_25(inputs)
        outputs_50 = quantile_model_50(inputs)
        outputs_75 = quantile_model_75(inputs)
        outputs_90 = quantile_model_90(inputs)

        predictions_mean.append(outputs_mean.cpu().numpy())  # Преобразуем обратно в NumPy массивы
        predictions_quantile_10.append(outputs_10.cpu().numpy())
        predictions_quantile_25.append(outputs_25.cpu().numpy())
        predictions_quantile_50.append(outputs_50.cpu().numpy())
        predictions_quantile_75.append(outputs_75.cpu().numpy())
        predictions_quantile_90.append(outputs_90.cpu().numpy())


predictions_mean = np.concatenate(predictions_mean, axis=0)  # Объединяем предсказания в один массив
predictions_10 = np.concatenate(predictions_quantile_10, axis=0)
predictions_25 = np.concatenate(predictions_quantile_25, axis=0)
predictions_50 = np.concatenate(predictions_quantile_50, axis=0)
predictions_75 = np.concatenate(predictions_quantile_75, axis=0)
predictions_90 = np.concatenate(predictions_quantile_90, axis=0)

merged_predictions = np.concatenate([
    predictions_mean,
    predictions_10,
    predictions_25,
    predictions_50,
    predictions_75,
    predictions_90
], axis=-1, dtype=np.float64)


print(merged_predictions.shape)

np.save('preds_100k_1.npy', merged_predictions)
target = np.load('часть2_19.npy')
print(target.shape)
mean = r2_score(target.reshape(-1, 1), predictions_mean.reshape(-1, 1))
q_10 = r2_score(target.reshape(-1, 1), predictions_10.reshape(-1, 1))
q_25 = r2_score(target.reshape(-1, 1), predictions_25.reshape(-1, 1))
q_50 = r2_score(target.reshape(-1, 1), predictions_50.reshape(-1, 1))
q_75 = r2_score(target.reshape(-1, 1), predictions_75.reshape(-1, 1))
q_90 = r2_score(target.reshape(-1, 1), predictions_90.reshape(-1, 1))

print((mean + q_10 + q_25 + q_50 + q_75 + q_90) / 6)

print(merged_predictions)

#print(f'Искомые величины {target[0, :, :]}')
#print(f'Предсказанные величины {predictions[0, :, :]}')
