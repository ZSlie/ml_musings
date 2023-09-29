import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transform
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import ImageTk, Image

import random
import time
import os
import copy

train_pd = pd.read_csv('../data/train.csv')

class MyDataset(Dataset):
  def __init__(self, features, labels, Transform):
    self.x = features
    self.y = labels
    self.transform = Transform

  def __len__(self):
    return len(self.x)

  def __getitem__(self, index):
    return self.transform(self.x[index]), self.y[index]
  
def GetDf(df, Transform):
  x_features = df.iloc[:, 1:].values
  y_labels = df.label.values
  x_features = x_features.reshape(-1, 1, 28, 28)
  x_features = np.uint8(x_features)
  x_features = torch.from_numpy(x_features)
  y_labels = torch.from_numpy(y_labels)
  return MyDataset(x_features, y_labels, Transform)

transformer = {
 '0': transform.Compose([
                           transform.ToPILImage(),
                           transform.Resize(94),
                           transform.Grayscale(num_output_channels=3), 
                           transform.ToTensor(),
                           transform.Normalize(
                                    [0.13097111880779266, 0.13097111880779266, 0.13097111880779266],
                                    [0.30848443508148193, 0.30848443508148193, 0.30848443508148193])
]),

    '1': transform.Compose([
                           transform.ToPILImage(),
                           transform.Resize(94),
                           transform.Grayscale(num_output_channels=3),
                           transform.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                           transform.RandomRotation(5),
                           transform.RandomAffine(degrees=11, translate=(0.1,0.1), scale=(0.8,0.8)),
                           transform.ToTensor(),
                           transform.Normalize(
                                    [0.13097111880779266, 0.13097111880779266, 0.13097111880779266],
                                    [0.30848443508148193, 0.30848443508148193, 0.30848443508148193])
]),
    'val': transform.Compose([
                           transform.ToPILImage(),
                           transform.Resize(94),
                           transform.Grayscale(num_output_channels=3),
                           transform.ToTensor(),
                           transform.Normalize(
                                  [0.13141274452209473, 0.13141274452209473, 0.13141274452209473],
                                  [0.30904173851013184, 0.30904173851013184, 0.30904173851013184])
    ])
}

exampleset = GetDf(train_pd, Transform=transform.Compose([
                           transform.ToPILImage(),
                           transform.Grayscale(num_output_channels=3),
                           transform.ToTensor()
    ]))

x, y = next(iter(torch.utils.data.DataLoader(exampleset)))

channels = ['Red', 'Green', 'Blue']
cmaps = [plt.cm.Reds_r, plt.cm.Greens_r, plt.cm.Blues_r]

fig, ax = plt.subplots(1, 4, figsize=(15, 10))

for i, axs in enumerate(fig.axes[:3]):
    axs.imshow(x[0][i,:,:], cmap=cmaps[i])
    axs.set_title(f'{channels[i]} Channel')
    axs.set_xticks([])
    axs.set_yticks([])
    
ax[3].imshow(x[0].permute(1,2,0), cmap='gray')
ax[3].set_title('Three Channels')
ax[3].set_xticks([])
ax[3].set_yticks([]);

class TestDataset(Dataset):
    def __init__(self, features,transform=transform.Compose([
                              transform.ToPILImage(),
        transform.Resize(94),
        transform.Grayscale(num_output_channels=3),
                              transform.ToTensor(),
                              transform.Normalize(
                                  [0.13141274452209473, 0.13141274452209473, 0.13141274452209473],
                                  [0.30904173851013184, 0.30904173851013184, 0.30904173851013184])
    ])):
        self.features = features.values.reshape((-1,28,28)).astype(np.uint8)
        self.targets = None
        self.transform=transform
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        return self.transform(self.features[idx])
    
def create_dataloaders(seed, test_size=0.1, df=train_pd, batch_size=50):    
  # Create training set and validation set
  train_data, valid_data = train_test_split(df,
                                            test_size=test_size,
                                            random_state=seed)

  # Create Datasets
  train_dataset_0 = GetDf(train_data, Transform=transformer['0'])
  train_dataset_1 = GetDf(train_data, Transform=transformer['1'])

  train_dataset = ConcatDataset([train_dataset_0, train_dataset_1])

  valid_dataset = GetDf(valid_data, Transform=transformer['val'])

  # Create Dataloaders
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
  valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4)

  train_size = len(train_dataset)
  val_size = len(valid_dataset)

  return train_loader, valid_loader, train_size, val_size

losses = {'train':[], 'val':[]}
accuracies = {'train':[], 'val':[]}

def train(seed, epochs, model):
  
  # Train and valid dataloaders
  print('Creating new dataloaders...')
    
  train_loader, valid_loader, train_size, val_size = create_dataloaders(seed=seed)
  
  loaders = {'train': train_loader, 'val': valid_loader}
  
  dataset_sizes = {'train': train_size, 'val': val_size}
  
  print('Creating a model {}...'.format(seed))
  # todo - ARGS
  use_cuda = torch.cuda.is_available() # not args.no_cuda and
  use_mps = torch.backends.mps.is_available() # not args.no_mps and
  print("Use CUDA: ", use_cuda)
  print("Use MPS: ", use_mps)
  
  # TODO: Seed
  # torch.manual_seed(args.seed)
  # TODO :: 
  #   if use_cuda:
  #     device = torch.device("cuda")
  #   elif use_mps:
  #     device = torch.device("mps")
  #   else:
  device = torch.device("cpu")
  
  model.to(device)  
  criterion = nn.CrossEntropyLoss()
  if seed==2 or seed==3:
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
  else:
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
  #   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
  
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, gamma=0.1)
  since = time.time()
  best_model = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  for epoch in range(epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()
      
      running_loss = 0.0
      running_corrects = 0.0
  
      for inputs, labels in loaders[phase]:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
  
        with torch.set_grad_enabled(phase=='train'):
          outp = model(inputs)
          _, pred = torch.max(outp, 1)
          loss = criterion(outp, labels)
        
          if phase == 'train':
            loss.backward()
            optimizer.step()
            
  
        running_loss += loss.item()*inputs.size(0)
        running_corrects += torch.sum(pred == labels.data)
  
  #       if phase == 'train':
  #           acc = 100. * running_corrects.double() / dataset_sizes[phase]
  #           scheduler.step(acc)
  
      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double()/dataset_sizes[phase]
      losses[phase].append(epoch_loss)
      accuracies[phase].append(epoch_acc)
      if phase == 'train':
        print('Epoch: {}/{}'.format(epoch+1, epochs))
      print('{} - loss:{}, accuracy{}'.format(phase, epoch_loss, epoch_acc))
    
      if phase == 'val':
        print('Time: {}m {}s'.format((time.time()- since)//60, (time.time()- since)%60))
        print('=='*31)
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = copy.deepcopy(model.state_dict())
    scheduler.step() 
  time_elapsed = time.time() - since
  print('CLASSIFIER TRAINING TIME {}m {}s'.format(time_elapsed//60, time_elapsed%60))
  print('=='*31)
  
  
  model.load_state_dict(best_model)
  
  for param in model.parameters():
        param.requires_grad=True
  
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  
  #   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, gamma=0.1)
  for epoch in range(epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()
      
      running_loss = 0.0
      running_corrects = 0.0
  
      for inputs, labels in loaders[phase]:
        inputs, labels = inputs.to(device), labels.to(device)
  
        optimizer.zero_grad()
  
        with torch.set_grad_enabled(phase=='train'):
          outp = model(inputs)
          _, pred = torch.max(outp, 1)
          loss = criterion(outp, labels)
        
          if phase == 'train':
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()*inputs.size(0)
        running_corrects += torch.sum(pred == labels.data)
  
  #       if phase == 'train':
  #         acc = 100. * running_corrects.double() / dataset_sizes[phase]
  #         scheduler.step(acc)
  
      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double()/dataset_sizes[phase]
      losses[phase].append(epoch_loss)
      accuracies[phase].append(epoch_acc)
      if phase == 'train':
        print('Epoch: {}/{}'.format(epoch+1, epochs))
      print('{} - loss:{}, accuracy{}'.format(phase, epoch_loss, epoch_acc))
    
      if phase == 'val':
        print('Time: {}m {}s'.format((time.time()- since)//60, (time.time()- since)%60))
        print('=='*31)    
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = copy.deepcopy(model.state_dict())
    scheduler.step() 
  time_elapsed = time.time() - since
  print('ALL NET TRAINING TIME {}m {}s'.format(time_elapsed//60, time_elapsed%60))
  print('=='*31)
  
  model.load_state_dict(best_model)
    
  model.eval() # Evaluation mode -> Turn off dropout
  test_pred = torch.LongTensor()
  
  if use_cuda:
    test_pred = test_pred.cuda()
        
  with torch.no_grad(): # Turn off gradients for prediction, saves memory and computations
    for features in test_loader:
        
        if use_cuda:
            features = features.cuda()
  
            # Get the softmax probabilities
            outputs = model(features)
            # Get the prediction of the batch
            _, predicted = torch.max(outputs, 1)
            # Concatenate the prediction
            test_pred = torch.cat((test_pred, predicted), dim=0)
    
  model_name = 'model_' + str(seed + 1)
  ensemble_df[model_name] = test_pred.cpu().numpy()
  print('Prediction Saved! \n') 

# TODO: fix warnings with: https://github.com/JaidedAI/EasyOCR/issues/766
densenet121_0 = torchvision.models.densenet121(pretrained=True)
for param in densenet121_0.parameters():
  param.requires_grad=False

densenet121_0.classifier = nn.Linear(in_features=densenet121_0.classifier.in_features, out_features=10, bias=True)

densenet121_1 = torchvision.models.densenet121(pretrained=True)
for param in densenet121_1.parameters():
  param.requires_grad=False

densenet121_1.classifier = nn.Linear(in_features=densenet121_1.classifier.in_features, out_features=10, bias=True)

googlenet = torchvision.models.googlenet(pretrained=True)
for param in googlenet.parameters():
  param.grad_requires = False

googlenet.fc = nn.Linear(in_features=googlenet.fc.in_features, out_features=10, bias=True)

resnet101 = torchvision.models.resnet101(pretrained=True)
for param in resnet101.parameters():
  param.grad_requires = False

resnet101.fc = nn.Linear(in_features=resnet101.fc.in_features, out_features=10, bias=True)

vgg19_bn = torchvision.models.vgg19_bn(pretrained=True)
for param in vgg19_bn.parameters():
  param.grad_requires = False

vgg19_bn.classifier[6] = nn.Linear(4096, 10, bias=True)

def main():
    # Create test_loader
    submit_df = pd.read_csv('../data/sample_submission.csv')
    test_df = pd.read_csv('../data/test.csv')
    test_dataset = TestDataset(test_df)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    ensemble_df = submit_df.copy()

    num_models = 5
    num_epochs = 10

    models = [densenet121_0, densenet121_1, googlenet, resnet101, vgg19_bn]

    for seed in range(num_models):
        train(seed=seed, epochs=num_epochs, model=models[seed])

if __name__ == '__main__':
  main()