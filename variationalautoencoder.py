import torch
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from torchvision import transforms


#Define constants.

TRAIN_DATA_SIZE = 60000
TEST_DATA_SIZE = 10000

DATA_SIZE = 28*28

BATCH_SIZE = 200

NUM_EPOCHS = 20

GENS_PER_DIGIT = 100

VARI_PARAMETER = 1

STD_MODIFIER = 1


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


#Declare the custom variational loss function.

def variation(mu, logvar):
    #p = torch.distributions.Normal(torch.zeros(mu.size()), torch.ones(logvar.size()))
    #q = torch.distributions.Normal(mu, logvar)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - (logvar).exp())
    #return torch.distributions.kl_divergence(p, q).mean()


#Declare the structure of the Autoencoder.

class VariationalAutoencoder(torch.nn.Module):
    
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 500),
            torch.nn.ReLU(),       
            torch.nn.Linear(500, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU()
        )
        self.to_mu = torch.nn.Sequential(
            torch.nn.Linear(100, 100)
        )
        self.to_std = torch.nn.Sequential(
            torch.nn.Linear(100, 100),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(), 
            torch.nn.Linear(200, 500),
            torch.nn.ReLU(), 
            torch.nn.Linear(500, 784),
            torch.nn.Sigmoid()
        )
        
    def sampling(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, input):
        enc = self.encoder(input)
        mu = self.to_mu(enc)
        std = self.to_std(enc)
        z = self.sampling(mu, std)
        return self.decoder(z), mu, std
    

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
    
model = VariationalAutoencoder()

opt = torch.optim.Adadelta(model.parameters())

mean_std_array = [] 
#10x2 of tensors, tensors are length 10, so effective dimension is 10x2x10
#1st dimension is the label for which distributions correspond to
#2nd dimension is mean / std
#3rd dimension is length of latent space

current_milli_time = lambda: int(round(time.time() * 1000))

before_time = current_milli_time()

for epoch in range(0, NUM_EPOCHS):
    batch_recon_loss = 0
    batch_vari_loss = 0
    for batch in range(0, int(TRAIN_DATA_SIZE/BATCH_SIZE)):
        input_tensor = torch.stack(data[0][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE])
        opt.zero_grad()
        output_tensor, mu, std = model(input_tensor.float())
        recon_loss = torch.nn.functional.binary_cross_entropy(output_tensor, input_tensor.float(), reduction="sum")
        vari_loss = variation(mu, std)
        loss = recon_loss + vari_loss*VARI_PARAMETER
        loss.backward()
        opt.step()
        batch_recon_loss += recon_loss.data.item()
        batch_vari_loss += vari_loss.data.item()
        if (batch == int(TRAIN_DATA_SIZE/BATCH_SIZE)-1):
            #print(output_tensor[0, :]*torch.tensor(256))
            #print(mu[0,:])
            #print(std[0,:])
            print(data[1][batch*BATCH_SIZE+epoch].item()*256)
            tensor = (output_tensor[epoch, :]*torch.tensor(256)).reshape(28, 28).int()
            image = tensor.clone().cpu()
            image = image.view(*tensor.size())
            image = transforms.ToPILImage()(image)
            plt.imshow(image)
            plt.show()
   
    print("")
    print("Epoch "+str(epoch+1)+"   Recon Loss : "+str(batch_recon_loss/(TRAIN_DATA_SIZE/BATCH_SIZE))+"   Variational Loss : "+str(batch_vari_loss/(TRAIN_DATA_SIZE/BATCH_SIZE)))
    
sorted_input_tensors = [[] for number in range(0, 10)]
for i in range(0, TRAIN_DATA_SIZE):
    label = int(data[1][i].item()*256)
    sorted_input_tensors[label].append(data[0][i])
sorted_input_tensors = [torch.stack(list) for list in sorted_input_tensors]
for input_tensor in sorted_input_tensors:
    latent_space = model.encoder.forward(input_tensor.float())
    b_mean = model.to_mu(latent_space)
    b_std = (0.5*model.to_std(latent_space)).exp()
    mean = torch.mean(b_mean, dim=0)
    std = torch.mean(b_std, dim=0)
    mean_std_array.append([mean, std])

after_time = current_milli_time()

seconds = math.floor((after_time-before_time)/1000)
minutes = math.floor(seconds/60)
seconds = seconds % 60
print(str(NUM_EPOCHS)+" epochs took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")    


#Generate new images.

image_file = open("VAE_GENERATED_IMAGES", "wb+")
label_file = open("VAE_GENERATED_LABELS", "wb+")

for number in range(0, 10):
    for i in range(0, GENS_PER_DIGIT):
        mean = mean_std_array[number][0]
        std = mean_std_array[number][1]
        gaussian = torch.randn(100)
        distribution = gaussian*std*torch.tensor(STD_MODIFIER)+mean
        image_tensor = model.decoder.forward(distribution.float())
        image_file.write(bytearray(list(map(int, (image_tensor*torch.tensor(256)).tolist()))))
        label_file.write(bytearray(int(number)))
    print("Images of "+str(number)+"s created.")

image_file.close()
label_file.close()

print("All images written.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    