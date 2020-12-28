from torch.utils.data import Dataset, DataLoader
import torch
import os
import cv2
import numpy as np
from torchvision import transforms

class GestureDataset(Dataset):
    def __init__(self, root, classes, per_n, filter_, transform):
        self.root = root
        self.classes = classes
        self.per_n = per_n
        self.transform = transform
        self.filter = filter_

    def imgs_in_cls(self, cls):
        return self.filter(os.listdir(os.path.join(self.root, cls))[::self.per_n])

    def __len__(self):
        length = 0
        for cls in self.classes:
            length += len(self.imgs_in_cls(cls))
        return length

    def __getitem__(self, idx):
        for i, cls in enumerate(self.classes):
            if idx < len(self.imgs_in_cls(cls)):
                img = cv2.imread(os.path.join(self.root, cls,
                                      self.imgs_in_cls(cls)[idx]))

                return self.transform(img), i
            idx -= len(self.imgs_in_cls(cls))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(0.8, 0.6, 0.4),
    transforms.RandomRotation(30, fill=(0,)),
    transforms.RandomResizedCrop((128,128), scale=(0.6, 1.2), ratio=(0.7, 1.3)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

def train_filter(images):
    return images[:int(len(images)*0.9)]

def test_filter(images):
    return images[int(len(images)*0.9):]

root = 'C:/datasets/asl_alphabet/asl_alphabet_train/'

def main():

    root = 'C:/datasets/asl_alphabet/asl_alphabet_train/'
    gestures = GestureDataset(root, os.listdir(root), 10, train_filter, transform)
    testset = GestureDataset(root, os.listdir(root), 10, test_filter, default_transform)
    loader = DataLoader(gestures, batch_size=8, shuffle=True, num_workers=6)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False)

    import torchvision.models as models
    import torch.nn as nn
    import torch.optim as optim

    #model = models.resnet18(pretrained=True).cuda()
    #model.fc = nn.Sequential(
    #    nn.Dropout(0.2),
    #    nn.Linear(512, 29)).cuda()

    model = models.mobilenet_v2(pretrained=True).cuda()
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 29)).cuda()

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    import wandb
    wandb.login()
    wandb.init(project='signlanguage',
              config={
                  'optimizer':'adam',
                  'learning_rate':1e-4,
                  'batch_size':8,
                  'network':'mobilenet(pretrained)'
              })
    wandb.watch(model)

    def run_epoch(model, loader, train, log_interval=50):
        if train:
            model.train()
        else:
            model.eval()

        epoch_loss = 0.
        interval_loss = 0.
        for i, (img, label) in enumerate(loader):
            img, label = img.float().cuda(), label.cuda()
            pred = model(img)
            loss = loss_func(pred, label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                interval_loss += loss.item()
                if (i+1)%log_interval == 0:
                    wandb.log({'interval_train_loss':interval_loss/log_interval})
                    interval_loss = 0.

            epoch_loss += loss.item()
        return epoch_loss / len(loader)

    min_loss = float('inf')
    for epoch in range(100):
        train_loss = run_epoch(model, loader, True)
        test_loss = run_epoch(model, test_loader, False)
        wandb.log({'train_loss':train_loss, 'test_loss':test_loss})

        torch.save(model.state_dict(), './checkpoints/mobilenet_chkpt{0}'.format(epoch))

if __name__=='__main__':
    main()
