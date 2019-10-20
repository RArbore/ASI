import torch
import numpy as np
import time
import math


#Define constants.

TRAIN_DATA_SIZE = 60000
TEST_DATA_SIZE = 10000

DATA_SIZE = 28*28

BATCH_SIZE = 200

NUM_EPOCHS = 1

GENS_PER_DIGIT = 10;

STD_MODIFIER = 0.5


#Read the MNIST dataset.

def read_mnist():
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
    return train_images_barray, train_labels_barray, test_images_barray, test_labels_barray


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

def arrange_data(train_images_barray, train_labels_barray, test_images_barray, test_labels_barray):
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
    return data
    
    
#Declare and train the network.
    
def train_model(data):
    model = Autoencoder()
    
    opt = torch.optim.Adadelta(model.parameters())
    
    #10x2 of tensors, tensors are length 10, so effective dimension is 10x2x10
    #1st dimension is the label for which distributions correspond to
    #2nd dimension is mean / std
    #3rd dimension is length of latent space
    
    current_milli_time = lambda: int(round(time.time() * 1000))
    
    before_time = current_milli_time()
    
    
    for epoch in range(0, NUM_EPOCHS):
        batch_loss = 0
        for batch in range(0, int(TRAIN_DATA_SIZE/BATCH_SIZE)):
            input_tensor = torch.stack(data[0][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE])
            opt.zero_grad()
            output_tensor = model(input_tensor.float())
            train_loss = torch.nn.functional.binary_cross_entropy(output_tensor, input_tensor.float(), reduction="sum")
            train_loss.backward()
            opt.step()
            batch_loss += train_loss.data.item()
                
        print("")
        print("Epoch "+str(epoch+1)+" Loss : "+str(batch_loss/(TRAIN_DATA_SIZE/BATCH_SIZE)))
        
    after_time = current_milli_time()
    
    seconds = math.floor((after_time-before_time)/1000)
    minutes = math.floor(seconds/60)
    seconds = seconds % 60
    
    print(str(NUM_EPOCHS)+" epochs took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")  
    return model;


#Generate new data.

def generate_data(model, data):
    mean_std_array = [] 
    sorted_input_tensors = [[] for number in range(0, 10)]
    for i in range(0, TRAIN_DATA_SIZE):
        label = int(data[1][i].item()*256)
        sorted_input_tensors[label].append(data[0][i])
    sorted_input_tensors = [torch.stack(list) for list in sorted_input_tensors]
    for input_tensor in sorted_input_tensors:
        latent_space = model.encoder.forward(input_tensor.float())
        mean = torch.mean(latent_space, dim=0)
        std = torch.std(latent_space, dim=0)
        mean_std_array.append([mean, std])
    return mean_std_array


#Generate new images.

def generate_images(model, mean_std_array):
    image_file = open("AE_GENERATED_IMAGES", "wb+")
    label_file = open("AE_GENERATED_LABELS", "wb+")
    
    for number in range(0, 10):
        for i in range(0, GENS_PER_DIGIT):
            mean = mean_std_array[number][0]
            std = mean_std_array[number][1]
            gaussian = torch.randn(10)
            distribution = gaussian*std*torch.tensor(STD_MODIFIER)+mean
            image_tensor = model.decoder.forward(distribution.float())
            image_file.write(bytearray(list(map(int, (image_tensor*torch.tensor(256)).tolist()))))
            label_file.write(bytearray(int(number)))
        print("Images of "+str(number)+"s created.")
    
    image_file.close()
    label_file.close()
    
    print("All images written.")
    
    
if __name__ == "__main__":
    train_images_barray, train_labels_barray, test_images_barray, test_labels_barray = read_mnist()
    data = arrange_data(train_images_barray, train_labels_barray, test_images_barray, test_labels_barray)
    model = train_model(data)
    mean_std_array = generate_data(model, data)
    generate_images(model, mean_std_array)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    