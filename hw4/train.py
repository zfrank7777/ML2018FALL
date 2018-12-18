import time
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from util import read_text, read_label, time_since, w2index
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        # self.fc0 = nn.Linear(input_size, hidden_size)
        self.gru = nn.LSTM(hidden_size, hidden_size,
                           num_layers=self.num_layers,
                           bidirectional=False, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * self.num_layers, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        # embed_input = self.fc0(input)
        embed_input = self.embedding(input.long())
        output, (hidden, _) = self.gru(embed_input)
        label = self.fc2(self.fc1(hidden))
        label = label.view(label.shape[1], -1)
        return label


class DcardDataset(Dataset):
    def __init__(self, data_np, train=True, label_np=None):
        self.data = torch.Tensor(data_np)
        if train:
            self.label = torch.from_numpy(label_np)
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.train:
            sample = {'data': self.data[idx],
                      'label': self.label[idx]}
        else:
            sample = {'data': self.data[idx]}

        return sample


train_text = read_text(sys.argv[1])
label = read_label(sys.argv[2])
# print (label.shape)
vec = w2index(train_text)
# print (vec.shape)
train_vec, valid_vec, train_label, valid_label = \
    train_test_split(vec, label, test_size=0.2, random_state=42)

""" HYPER-PARAMETESRS """
input_size = 7747
batch_size = 512
n_epochs = 10
print_every = 1
hidden_size = 96
lr = 0.0005

""" DATASET AND LOADERS """
train_dataset = DcardDataset(data_np=train_vec,
                             label_np=train_label,
                             train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = DcardDataset(data_np=valid_vec,
                             label_np=valid_label,
                             train=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

""" MODEL AND LOSS """
model = RNN(input_size, hidden_size, batch_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
for epoch in range(1, n_epochs + 1):
    epoch_loss = 0
    epoch_acc = 0
    for i, sample in enumerate(train_loader):
        x = sample['data'].to(device)
        label = sample['label'].to(device)
        optimizer.zero_grad()
        output_label = model(x)
        loss = criterion(output_label, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss
        _, preds = torch.max(output_label.data, 1)
        epoch_acc += torch.sum(preds == label)

    if epoch % print_every == 0:
        with torch.no_grad():
            valid_acc = 0
            valid_loss = 0
            for i, sample in enumerate(valid_loader):
                x = sample['data'].to(device)
                label = sample['label'].to(device)
                optimizer.zero_grad()
                output_label = model(x)
                loss = criterion(output_label, label)
                _, preds = torch.max(output_label.data, 1)
                valid_loss += criterion(output_label, label)
                valid_acc += torch.sum(preds == label)

        print('[%s (%d %d%%), Loss:  %.3f, train_Acc: %.3f, valid_Loss: %.3f, valid_Acc: %.3f]' %
              (time_since(start),
               epoch,
               epoch / n_epochs * 100,
               epoch_loss/len(train_loader),
               float(epoch_acc) / len(train_loader) / batch_size,
               valid_loss/len(valid_loader),
               float(valid_acc) / len(valid_loader) / batch_size))
        epoch_loss = epoch_acc = 0
if torch.cuda.is_available():
    torch.save(model.state_dict(), "rnn_gpu")
else:
    torch.save(model.state_dict(), "rnn_cpu")
