#Don't change batch size
batch_size = 64

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms


## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST('/data/patrick-data/pytorch/data/',
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
test_data = datasets.MNIST('/data/patrick-data/pytorch/data/',
                           train=False,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
subset_indices = ((train_data.train_labels == 0) +
                  (train_data.train_labels == 1)).nonzero()
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=False,
    sampler=SubsetRandomSampler(subset_indices))


subset_indices = ((test_data.test_labels == 0) +
                  (test_data.test_labels == 1)).nonzero()
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False,
    sampler=SubsetRandomSampler(subset_indices))


# Training the Model
# Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.

for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        #Convert labels from 0,1 to -1,1
        labels = Variable(2*(labels.float()-0.5))

        ## TODO

    # Print your results every epoch


# Test the Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))

    ## Put your prediction code here

    correct += (prediction.view(-1).long() == labels).sum()
    total += images.shape[0]
print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))
