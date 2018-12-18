import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from util import read_text, w2v

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.fc0 = nn.Linear(input_size, hidden_size)
        self.gru = nn.LSTM(hidden_size, hidden_size,
                           num_layers=self.num_layers,
                           bidirectional=False, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * self.num_layers, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        embed_input = self.fc0(input)
        output, (hidden, _) = self.gru(embed_input)
        label = self.fc2(self.fc1(hidden))
        label = label.view(label.shape[1], -1)
        return label


class DcardDataset(Dataset):
    def __init__(self, data_np, train=True, label_np=None):
        self.data = torch.Tensor(data_np)
        if train and label_np:
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


""" HYPER-PARAMETESRS """
input_size = 128
batch_size = 1
hidden_size = 96

""" DATASET AND LOADERS """
test_text = read_text(sys.argv[1])
UNK = np.random.random(128)
EOS = np.random.random(128)
test_vec = w2v(test_text, UNK, EOS)
test_dataset = DcardDataset(data_np=test_vec,
                            train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

""" MODEL AND LOSS """
model1 = RNN(input_size, hidden_size, batch_size).to(device)
if torch.cuda.is_available():
    model1.load_state_dict(torch.load("rnn_gpu"))
else:
    model1.load_state_dict(torch.load("rnn_cpu"))
model1.eval()
f = open(sys.argv[2], "w")
f.write('id,label\n')
for i, sample in enumerate(test_loader):
    x = sample['data'].to(device)
    output_label = model1(x)
    _, preds = torch.max(output_label.data, 1)
    preds = preds.to(torch.device('cpu'))
    preds1 = int(preds.numpy())
    f.write(str(i)+','+str(preds1)+'\n')
f.close()
