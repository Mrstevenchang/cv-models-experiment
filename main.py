import torch
import torch.nn as nn
from models.mae import MAE_Encoder
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import *
from torch.utils.data import DataLoader


train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
train_dataloader = DataLoader(train_dataset, 10, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mae_encoder = MAE_Encoder()

mae_encoder.to(device)

for img, label in iter(train_dataloader):
    img = img.to(device)
    labels = label.to(device)
    img = mae_encoder(img)
    print(img)


