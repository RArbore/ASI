import torch
import numpy as np
import time
import math


#Define constants.

TRAIN_DATA_SIZE = 60000
TEST_DATA_SIZE = 10000

DATA_SIZE = 28*28

BATCH_SIZE = 200

GENS = 1000

MIN_EPOCHS = 50

ABSOLUTE_EPOCHS = 150

LABEL_SMOOTHING = 0.9


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


#Declare the structure of the Discriminator.

class Discriminator(torch.nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, (17, 9)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(6, 11, (16, 5)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(11, 16, (5, 3)),
            torch.nn.LeakyReLU(0.2),
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(16, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 3),
            torch.nn.Tanh(),
            torch.nn.Linear(3, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, input):
        out = input.view(input.size(0), 1, 28*2, 28)
        out = self.conv(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


#Declare the structure of the Generator.

class Generator(torch.nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.tconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(10, 8, 6),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 6, 6),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(6, 4, 6),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(4, 2, 7),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(2, 1, 7),
            torch.nn.Sigmoid()
        )
    
    def forward(self, input):
        out = input.view(1, 10, 1, 1)
        out = self.tconv(out)
        out = out.view(DATA_SIZE)
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
    discriminator = Discriminator()
    generator = Generator()
    
    discriminator_opt = torch.optim.Adadelta(discriminator.parameters(), lr=4.0)
    generator_opt = torch.optim.Adadelta(generator.parameters())
    
    current_milli_time = lambda: int(round(time.time() * 1000))
    
    before_time = current_milli_time()
    
    discriminator_incorrect = 0
    
    epoch = 0
    
    while (discriminator_incorrect < 0.4 or epoch < MIN_EPOCHS) and  epoch < ABSOLUTE_EPOCHS:
        discriminator_batch_loss = 0
        generator_batch_loss = 0
        for batch in range(0, int(TRAIN_DATA_SIZE/BATCH_SIZE)):
            discriminator_opt.zero_grad()
            generator_opt.zero_grad()
            real_images = data[0][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            fake_images = [generator.forward(torch.rand(10)).double() for i in range(0, BATCH_SIZE)]
            input_tensor = torch.stack(real_images+fake_images)
            doubled_up_input = input_tensor.view(BATCH_SIZE, DATA_SIZE*2)
            doubled_up_input += (torch.randn(doubled_up_input.size())/5).double()
            decision = discriminator(doubled_up_input.float())
            decision = decision.view(-1)
            
            discriminator_label = torch.tensor([LABEL_SMOOTHING]*int(BATCH_SIZE/2)+[0]*int(BATCH_SIZE/2))
            discriminator_incorrect = 2*(torch.sum(torch.abs(decision-discriminator_label.float()))/(2*BATCH_SIZE)).item()
            discriminator_train_loss = torch.nn.functional.binary_cross_entropy(decision, discriminator_label.float(), reduction="sum")
            discriminator_train_loss.backward()
            discriminator_opt.step()
            discriminator_batch_loss += discriminator_train_loss.data.item()
            
            generated = torch.stack([generator.forward(torch.rand(10)).double() for i in range(0, BATCH_SIZE)]).float()
            generator_pred = discriminator(generated.view(int(BATCH_SIZE/2), DATA_SIZE*2))
            generator_label = torch.ones(int(BATCH_SIZE/2))
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
    return discriminator, generator;


#Generate images.

def generate_images(generator):
    image_file = open("DGAN_GENERATED_IMAGES", "wb+")
    
    for i in range(GENS):
        image_tensor = generator.forward(torch.rand(10))
        image_tensor = image_tensor*torch.tensor(256)
        image_tensor = torch.min(image_tensor, (torch.ones(image_tensor.size())*255).float())
        image_tensor = torch.max(image_tensor, (torch.zeros(image_tensor.size())).float())
        image_file.write(bytearray(list(map(int, (image_tensor.tolist())))))
    image_file.close()
    
    
if __name__ == "__main__":
    train_images_barray, train_labels_barray, test_images_barray, test_labels_barray = read_mnist()
    data = arrange_data(train_images_barray, train_labels_barray, test_images_barray, test_labels_barray)
    generator, discriminator = train_model(data)
    generate_images(generator)
    
    
    
    
    
    