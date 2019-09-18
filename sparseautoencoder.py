import torch
import numpy as np
import time
import math


#Define constants.

TRAIN_DATA_SIZE = 60000
TEST_DATA_SIZE = 10000

DATA_SIZE = 28*28

BATCH_SIZE = 200

NUM_EPOCHS = 20

GENS_PER_IMAGE = 2

SPARSITY = 0.05

SPARSE_MODIFIER = 20


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
        ba.append(int.from_bytes(byte, byteorder="big")/256)
        byte = f.read(1)
for i in range(0, 16):
    train_images_barray.pop(0)
    test_images_barray.pop(0)
for i in range(0, 8):
    train_labels_barray.pop(0)
    test_labels_barray.pop(0)
print("MNIST loaded.")


#Declare the custom sparse loss function.

def sparsity(ae, batch):
    latent = model.encoder.forward(batch.float())
    p_hat = torch.mean(latent, 0)
    p = SPARSITY
    sum = 0
    for j in range(0, 10):
        sum += p*torch.log(p/(p_hat[j]))+(1-p)*torch.log((1-p)/(1-(p_hat[j])))
    return sum
    

#Declare the structure of the Autoencoder.

class SparseAutoencoder(torch.nn.Module):
    
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
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
            torch.nn.Sigmoid()
        )
    
    def forward(self, input):
        latent = self.encoder(input)
        out = self.decoder(latent)
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
    
model = SparseAutoencoder()

loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adadelta(model.parameters())

current_milli_time = lambda: int(round(time.time() * 1000))

before_time = current_milli_time()

for epoch in range(0, NUM_EPOCHS):
    batch_recon_loss = 0
    batch_sparse_loss = 0
    for batch in range(0, int(TRAIN_DATA_SIZE/BATCH_SIZE)):
        input_tensor = torch.stack(data[0][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE])
        opt.zero_grad()
        output_tensor = model(input_tensor.float())
        recon_loss = loss_fn(output_tensor, input_tensor.float())
        sparse_loss = sparsity(model, input_tensor.float())*SPARSE_MODIFIER
        loss = recon_loss + sparse_loss
        loss.backward()
        opt.step()
        batch_recon_loss += recon_loss.data.item()
        batch_sparse_loss += sparse_loss.data.item()
    print("")
    print("Epoch "+str(epoch+1)+"   Recon Loss : "+str(batch_recon_loss/(TRAIN_DATA_SIZE/BATCH_SIZE))+"   Sparse Loss : "+str(batch_sparse_loss/(TRAIN_DATA_SIZE/BATCH_SIZE)))
    

after_time = current_milli_time()

seconds = math.floor((after_time-before_time)/1000)
minutes = math.floor(seconds/60)
seconds = seconds % 60
print(str(NUM_EPOCHS)+" epochs took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")    


#Generate new images.

image_file = open("SAE_GENERATED_IMAGES", "wb+")
label_file = open("SAE_GENERATED_LABELS", "wb+")

count = 0

mu, sigma = 0, 0.1

for inimage, label in zip(data[0], data[1]):
    latent_space_tensor = model.encoder.forward(inimage.float())
    latent_space = latent_space_tensor.tolist()
    for num in range(0, GENS_PER_IMAGE):
        new_latent_space = [i for i in latent_space]
        for i in range(0, len(latent_space)):
            new_latent_space[i] = new_latent_space[i]+np.random.normal(mu, sigma, 1)
        new_latent_tensor = torch.tensor(new_latent_space).float().view(-1)
        image_tensor = model.decoder.forward(new_latent_tensor)
        image_file.write(bytearray(list(map(int, (image_tensor*torch.tensor(256)).tolist()))))
        label_file.write(bytearray(int(label.tolist())))
    count += 1
    if (count % 1000 == 0):
        print("1000 images looped through.")
image_file.close()
label_file.close()
print("Images written.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    