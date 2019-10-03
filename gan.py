import torch
import numpy as np
import time
import math

#Define constants.

TRAIN_DATA_SIZE = 60000
TEST_DATA_SIZE = 10000

DATA_SIZE = 28*28

BATCH_SIZE = 200

GENS = 10

MIN_EPOCHS = 25


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
    increment = 0
    while byte:
        ba.append(int.from_bytes(byte, byteorder="big")/256)
        increment += 1
        byte = f.read(1)
    print("*", end="")
for i in range(0, 16):
    train_images_barray.pop(0)
    test_images_barray.pop(0)
for i in range(0, 8):
    train_labels_barray.pop(0)
    test_labels_barray.pop(0)
print("")
print("MNIST loaded.")


#Declare the structure of the Discriminator.

class Discriminator(torch.nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(784, 300),
            torch.nn.ReLU(),       
            torch.nn.Linear(300, 100),
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 40),
            torch.nn.ReLU(), 
            torch.nn.Linear(40, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, input):
        out = self.linear(input)
        return out


#Declare the structure of the Generator.

class Generator(torch.nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(10, 40),
            torch.nn.ReLU(), 
            torch.nn.Linear(40, 100),
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 300),
            torch.nn.ReLU(), 
            torch.nn.Linear(300, 784),
            torch.nn.Sigmoid()
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
    
discriminator = Discriminator()
generator = Generator()

discriminator_opt = torch.optim.Adadelta(discriminator.parameters())
generator_opt = torch.optim.Adadelta(generator.parameters())

current_milli_time = lambda: int(round(time.time() * 1000))

before_time = current_milli_time()

discriminator_incorrect = 0

epoch = 0

while discriminator_incorrect < 0.4 or epoch < MIN_EPOCHS:
    discriminator_batch_loss = 0
    generator_batch_loss = 0
    for batch in range(0, int(TRAIN_DATA_SIZE/BATCH_SIZE)):
        discriminator_opt.zero_grad()
        generator_opt.zero_grad()
        real_images = data[0][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
        fake_images = [generator.forward(torch.rand(10)).double() for i in range(0, BATCH_SIZE)]
        input_tensor = torch.stack(real_images+fake_images)
        decision = discriminator(input_tensor.float())
        decision = decision.resize(decision.size()[0])
        
        discriminator_label = torch.tensor([1]*BATCH_SIZE+[0]*BATCH_SIZE)
        discriminator_incorrect = (torch.sum(torch.abs(decision-discriminator_label.float()))/(2*BATCH_SIZE)).item()
        discriminator_train_loss = torch.nn.functional.binary_cross_entropy(decision, discriminator_label.float(), reduction="sum")
        discriminator_train_loss.backward()
        discriminator_opt.step()
        discriminator_batch_loss += discriminator_train_loss.data.item()
        
        generated = torch.stack([generator.forward(torch.rand(10)).double() for i in range(0, BATCH_SIZE)]).float()
        generator_pred = discriminator(generated)
        generator_label = torch.ones(BATCH_SIZE)
        generator_train_loss = torch.nn.functional.binary_cross_entropy(generator_pred.resize(generator_pred.size()[0]), generator_label.float(), reduction="sum")
        generator_train_loss.backward()
        generator_opt.step()
        generator_batch_loss += generator_train_loss.item()
        if ((batch+1)%int((TRAIN_DATA_SIZE/BATCH_SIZE)/10) == 0):
            print("DBL : "+str(discriminator_batch_loss/(batch+1))+"   GBL : "+str(generator_batch_loss/(batch+1)))
    print("")
    print("Discriminator Epoch "+str(epoch+1)+" Loss : "+str(discriminator_batch_loss/(TRAIN_DATA_SIZE/BATCH_SIZE)))
    print("Generator Epoch "+str(epoch+1)+" Loss : "+str(generator_batch_loss/(TRAIN_DATA_SIZE/BATCH_SIZE)))
    print("Portion Discriminator Incorrect : "+str(discriminator_incorrect))
    print("")
    epoch += 1

after_time = current_milli_time()

seconds = math.floor((after_time-before_time)/1000)
minutes = math.floor(seconds/60)
seconds = seconds % 60
print(str(epoch)+" epochs took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")  