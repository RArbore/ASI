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

for f, ba in zip(files, barrays):
    byte = f.read(1)
    while byte:
        ba.append(int.from_bytes(byte, byteorder="big"))
        byte = f.read(1)
for i in range(0, 16):
    train_images_barray.pop(0)
    test_images_barray.pop(0)
for i in range(0, 8):
    train_labels_barray.pop(0)
    test_labels_barray.pop(0)
print("MNIST loaded.")


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
        out = self.encoder(input)
        out = self.decoder(out)
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
            t_array.append(torch.from_numpy(np.asarray(train_images_barray[i*784:(i+1)*784])))
    elif (i == 1):
        for i in range(0, TRAIN_DATA_SIZE):
            t_array.append(torch.from_numpy(np.asarray(train_labels_barray[i])))
    elif (i == 2):
        for i in range(0, TEST_DATA_SIZE):
            t_array.append(torch.from_numpy(np.asarray(test_images_barray[i*784:(i+1)*784])))
    elif (i == 3):
        for i in range(0, TEST_DATA_SIZE):
            t_array.append(torch.from_numpy(np.asarray(test_labels_barray[i])))
    data.append(t_array)
print("Data arranged.")
    
    
#Declare and train the network.
    
model = Autoencoder()

loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adadelta(model.parameters())

current_milli_time = lambda: int(round(time.time() * 1000))

before_time = current_milli_time()

for epoch in range(0, NUM_EPOCHS):
    batch_loss = 0
    correct = 0
    incorrect = 0
    for batch in range(0, int(TRAIN_DATA_SIZE/BATCH_SIZE)):
        input_tensor = torch.stack(data[0][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE])
        label_tensor = torch.stack(data[1][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE])
        opt.zero_grad()
        output_tensor = model(input_tensor.float())
        train_loss = loss_fn(output_tensor, label_tensor.long())
        train_loss.backward()
        opt.step()
        batch_loss += train_loss.data.item()
        preds = torch.max(output_tensor, 1).indices
        for p, l in zip(preds.tolist(), label_tensor.tolist()):
            if p == l:
                correct += 1
            else:
                incorrect += 1
    print("Epoch Loss : "+str(batch_loss/(TRAIN_DATA_SIZE/BATCH_SIZE)))
    print("Percent Correct : "+str(correct/(correct+incorrect)*100.0)+"%")
    print("")

after_time = current_milli_time()

seconds = math.floor((after_time-before_time)/1000)
minutes = math.floor(seconds/60)
seconds = seconds % 60
print(str(NUM_EPOCHS)+" epochs took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    