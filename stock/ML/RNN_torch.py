import pandas as pd
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Defining the layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:,-1,:])
        # h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        # c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        # out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out = self.fc(out[:, -1, :])
        return out, hidden


def load_data(index_name):
    curPath = os.path.abspath(os.path.dirname(__file__))
    name = '/Indices data/' + index_name + '.csv'
    Path = os.path.abspath(curPath + name)
    data = pd.read_csv(Path)
    data_idx = data.set_index(["Date"], drop=True)
    date = data_idx.index
    close_index = data_idx['Close']
    return close_index, date


def create_dataset(dataset, seq=5):
    dataX, dataY = [], []
    for i in range(len(dataset)- seq):
        ls = dataset[i:(i+seq)]
        dataX.append(ls)
        dataY.append(dataset[i + seq])
    return dataX, dataY


def prepare_data(dataset, split_rate=0.8):
    train_size = int(len(dataset) * split_rate)
    train_data, test_data = dataset[: train_size], dataset[train_size:]
    train_X, train_Y = create_dataset(train_data, seq=5)
    test_X, test_Y = create_dataset(test_data, seq=5)
    return train_X,train_Y,test_X,test_Y


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for data, target in train_loader:
        data, target = data.to(torch.float32).to(device), target.to(torch.float32).to(device)
        data = data.view(-1, 5, 1)
        target = target.view(-1, 1)
        optimizer.zero_grad()
        output, _ = model(data)
        loss_fn = nn.MSELoss()
        loss_ = loss_fn(output, target)
        loss_.backward()
        optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss_.item()))


def model_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(torch.float32).to(device), target.to(torch.float32).to(device)
            data = data.view(-1, 5, 1)
            target = target.view(-1, 1)
            output,_ = model(data)
            test_loss += F.mse_loss(output, target)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))


def normalize(data):
    dataset = np.reshape(data, (-1, 1))
    norm = MinMaxScaler(feature_range=(0, 1))
    dataset = norm.fit_transform(dataset)
    return dataset, norm


def predict(data, date, model, device, seq=5):
    predict_data, norm = normalize(data)
    date = np.array(date)
    new = date[-1]
    new_date = []
    for i in range(5):
        new_ = datetime.datetime.strptime(new, "%Y-%m-%d").date()
        new_ = new_ + datetime.timedelta(days=i+1)
        new_date.append(str(new_))
    for i in range(seq):
        model.eval()
        predict_data = torch.from_numpy(predict_data)
        with torch.no_grad():
            predict_data = predict_data.to(torch.float32).to(device)
            data = predict_data.view(-1, 5, 1)
            output, _ = model(data)
            predict_data = np.append(predict_data[1:], output)
    predict_data = np.reshape(predict_data, (-1, 1))
    predict_data = norm.inverse_transform(predict_data)
    predict_data = pd.DataFrame(predict_data, new_date)
    return predict_data


def train_model(data, index_name):
    dataset, norm = normalize(data.values)
    train_data, train_target, test_data, test_target = prepare_data(dataset)
    train_dataset = Dataset(train_data, train_target)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = Dataset(test_data, test_target)
    test_loader = torch.utils.data.DataLoader(test_dataset)
    device = torch.device('cpu')
    torch.manual_seed(1)
    model = Model(input_size=1, output_size=1, hidden_dim=16, n_layers=1).to(device)
    epochs = 50
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    curPath = os.path.abspath(os.path.dirname(__file__))
    name = '/model/' + index_name + '.pt'
    Path = os.path.abspath(curPath + name)
    base_path = "{}".format(Path)
    if os.path.exists(base_path):
        model.load_state_dict(torch.load(base_path))
    else:
        for epoch in range(1,epochs+1):
            train(model, device, train_loader, optimizer, epoch)
            model_test(model, device, test_loader)
        torch.save(model.state_dict(), base_path)
    return model, device


def plot_origin_predict(index_name):
    close_index, date = load_data(index_name)
    model, device = train_model(close_index, index_name)
    all_test_data, norm = normalize(close_index.values)
    all_test_data, _ = create_dataset(all_test_data)
    result = []
    for data in all_test_data:
        model.eval()
        data = torch.from_numpy(data)
        with torch.no_grad():
            data = data.to(torch.float32).to(device)
            data = data.view(-1, 5, 1)
            output, _ = model(data)
            result.append(output)
    result = np.reshape(result, (-1, 1))
    result = norm.inverse_transform(result)
    result = np.reshape(result, -1)
    # result = pd.DataFrame(result)
    plt.plot(close_index.values, color='y')
    plt.plot(result, color='b')
    return close_index.values, result


def main(index_name):
    close_index, date = load_data(index_name)
    model, device = train_model(close_index, index_name)
    predict_data = (close_index.values)[-5:]
    predict_result = predict(predict_data, date, model, device)
    return predict_result

if __name__ == '__main__':
    predict_result = main('^SZ')
    file = ['^SZ', '^Ixic','^Rut', '^Hsi']
    for name in file:
        close_, test = plot_origin_predict(name)
        plt.plot(close_, color='y')
        plt.plot(test, color='b')
        plt.show()
    # plt.plot(predict_result)
    # plt.show()