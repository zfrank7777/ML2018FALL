import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image
import sys


# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input channels = 1, output channels = 10
        self.conv1 = torch.nn.Conv2d(
                1, 16, kernel_size=5, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(
                16, 32, kernel_size=5, stride=1, padding=1)
        self.conv25 = torch.nn.Conv2d(
                16, 64, kernel_size=5, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.norm16 = torch.nn.BatchNorm2d(16)
        self.norm32 = torch.nn.BatchNorm2d(32)
        self.norm64 = torch.nn.BatchNorm2d(64)
        self.norm128 = torch.nn.BatchNorm2d(128)
        self.norm256 = torch.nn.BatchNorm2d(256)
        self.norm512 = torch.nn.BatchNorm2d(512)
        self.leaky = torch.nn.LeakyReLU(0.2)
        self.drop = torch.nn.Dropout(p=0.3)
        self.dropfc = torch.nn.Dropout(p=0.5)
        self.fc1 = torch.nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 7)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 48, 48)
        x = F.relu(self.norm16(self.conv1(x)))
        x = self.leaky(self.norm32(self.conv2(x)))
        x = self.drop(x)
        x = self.leaky(self.norm64(self.conv3(x)))
        x = self.drop(self.pool(x))
        x = self.leaky(self.norm128(self.conv4(x)))
        x = self.drop(self.pool(x))
        x = self.leaky(self.norm256(self.conv5(x)))
        x = self.drop(self.pool(x))
        x = self.leaky(self.norm512(self.conv6(x)))
        x = self.drop(self.pool(x))

        x = x.view(-1, 512 * 2 * 2)

        x = self.fc1(x)
        x = self.leaky(x)
        x = self.dropfc(x)
        x = self.fc2(x)
        x = self.leaky(x)
        x = self.fc3(x)
        return(x)


def train_model(batch_size, n_epochs):
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            train = Variable(images)
            train = train.type(torch.FloatTensor).to(device)
            labels = Variable(labels).to(device)
            optimizer.zero_grad()
            outputs = model(train)
            loss = error(outputs, labels)
            loss.backward()
            optimizer.step()
            count += 1
            if count % 100 == 0:
                correct = 0
                total = 0
                for images, labels in train_loader:
                    images = Variable(images)
                    images = images.type(torch.FloatTensor).to(device)
                    labels = Variable(labels).to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                train_acc = 100 * correct / total

                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = Variable(images)
                    images = images.type(torch.FloatTensor).to(device)
                    labels = Variable(labels).to(device)
                    outputs = model(images)
                    val_loss = error(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total

                # Print Loss
                L = loss.data.to(torch.device("cpu")).numpy()
                val_loss = val_loss.to(torch.device("cpu")).detach().numpy()

    return loss_list, iteration_list, accuracy_list


# Transforms
mytransform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.1,0.1),
        scale=(0.9,1.1),
        shear=0.1,
        resample=False,
        fillcolor=0),
    transforms.ToTensor()
])
totensor = transforms.ToTensor()


#get dataset
class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self,df,loader=torchvision.datasets.folder.default_loader,train=False,transform=None):
        self.df = df
        self.loader = loader
        self.train = train
        self.transform = transform
        if train is True:
            self.label = np.array(df['label'])
        self.df = np.array(df['feature'].str.split(' ').values.tolist()).reshape(-1,48,48).astype(np.float)
    def __getitem__(self,index):
        img = self.df[index].astype('uint8')
        img = Image.fromarray(img,'L')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = totensor(img)
        if self.train is True:
            target = self.label[index]
            return img,target
        else:
            return img
    def __len__(self):
        n = len(self.df)
        return n


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print (device)
# batch_size, epoch and iteration
batch_size = 512 if torch.cuda.is_available() else 32
num_epochs = 400

fetch = pd.read_csv(sys.argv[1])
split = int(0.2*len(fetch))
train_dataset = ImagesDataset(df=fetch,train=True,transform=mytransform)
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

test_dataset = ImagesDataset(df=fetch[:split],train=True)
test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)


model = CNN()
model.to(device)
error = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_list, iteration_list, accuracy_list =  \
        train_model(batch_size, num_epochs)

if torch.cuda.is_available():
    torch.save(model.state_dict(), "model_gpu")
else:
    torch.save(model.state_dict(), "model_cpu")
