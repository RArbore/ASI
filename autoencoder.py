import torch
import numpy as np
import time
import math


#Define constants.

TRAIN_DATA_SIZE = 60000
TEST_DATA_SIZE = 10000

DATA_SIZE = 28*28

BATCH_SIZE = 100

NUM_EPOCHS = 100


#Read the MNIST dataset.

train_images_file = open("MNIST_TRAIN_IMAGES", "rb")
train_labels_file = open("MNIST_TRAIN_LABELS", "rb")
test_images_file = open("MNIST_TEST_IMAGES", "rb")
test_labels_file = open("MNIST_TEST_LABELS", "rb")

train_images_barray = []
train_labels_barray = []
test_images_barray = []
test_labels_barray = []

files = [train_images_file, train_labels_file, test_images_file, test_labels_file]
barrays = [train_images_barray, train_labels_barray, test_images_barray, test_labels_barray]

i = 0
for f, ba in zip(files, barrays):
    byte = f.read(1)
    while byte:
        if (i >= 16):
            ba.append(byte)
        byte = f.read(1)
        i+=1

print(len(train_images_barray))

#Declare the structure of the Autoencoder.

class Autoencoder(torch.nn.Module):
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 300),
            torch.nn.ReLU(),       
            torch.nn.Linear(300, 100),
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 40),
            torch.nn.ReLU(), 
            torch.nn.Linear(40, 10),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 40),
            torch.nn.ReLU(), 
            torch.nn.Linear(40, 100),
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 300),
            torch.nn.ReLU(), 
            torch.nn.Linear(300, 784),
            torch.nn.ReLU()
        )
    
    def forward(self, input):
        out = self.linear(input)
        return out
    

'''
Arrange the MNIST data into 4 arrays of tensors, 2 for the training data and 2 for the test data.
For the pair of arrays for training and testing, 1 is for the images and 1 is for the labels. 
In the image array, each tensor is 784 elements, and the array is 60000 elemenets. 
In the label array, each tensor is a single element, and the array is 60000 elements.        
'''

data = []
for i in range(0, 4):
    t_array = []
    if (i == 0):
        for i in range(0, TRAIN_DATA_SIZE):
            t_array.append(torch.from_numpy(np.asarray(train_images_barray[i*784, (i+1)*784])))
    elif (i == 1):
        for i in range(0, TRAIN_DATA_SIZE):
            print(train_labels_barray[i])
            t_array.append(torch.from_numpy(np.asarray(train_labels_barray[i])))
    data.append(t_array)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    