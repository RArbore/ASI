import torch
from torchvision import transforms
import time
import math
import os
import gc

#Define constants.

DATA_SIZE = 149*173*192

B_DATA_DIMENSIONS = [149, 173, 192]

DATA_DIMENSIONS = [6, 6, 6]

TRAIN_DATA_SIZE = 335

BATCH_SIZE = 5

MIN_EPOCHS = 50

ABSOLUTE_EPOCHS = 100

NUM_GENERATED_PER_EPOCH = 10

LABEL_NOISE = 0.25

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

folder = ""

#torch.manual_seed(0)


class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(100, 1, (6, 6, 6), bias=False),
            torch.nn.Tanh(),
        )
        for layer in self.conv:
            if "ConvTranspose3d" in str(layer):
                torch.nn.init.xavier_uniform_(layer.weight)
                #torch.nn.init.xavier_uniform_(layer.bias)

    def forward(self, input, epoch=-1, batch=-1):
        out = input.view(-1, 100, 1, 1, 1)
                
        out = self.conv(out)
        
        return (out+1)/2 * torch.tensor(0.998) + torch.tensor(0.001)

class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(1, 3, (2, 2, 2)),
            torch.nn.Tanh(),
            torch.nn.Conv3d(3, 6, (2, 2, 2)),
            torch.nn.Tanh(),
            torch.nn.Conv3d(6, 8, (2, 2, 2)),
            torch.nn.Tanh(),
            torch.nn.Conv3d(8, 10, (2, 2, 2)),
            torch.nn.Tanh(),
            torch.nn.Conv3d(10, 12, (2, 2, 2)),
            torch.nn.Tanh(),
        )
        
        self.acti = torch.nn.Sequential(
            torch.nn.Linear(12, 4),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(4, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, input):
        out = input
                
        out = self.conv(out)
        out = self.acti(out.view(input.size(0), 12))
        
        return out.view(input.size(0), 1) * torch.tensor(0.998) + torch.tensor(0.001)
    
def save_image(tensor, filename):
    ndarr = tensor.mul(256).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)
    
def current_milli_time():
    return int(round(time.time() * 1000))

def train_model(data):
    discriminator = Discriminator().to(device).half()
    generator = Generator().to(device).half()

    discriminator_opt = torch.optim.Adam(discriminator.parameters(), eps=0.0001)
    generator_opt = torch.optim.Adam(generator.parameters(), eps=0.0001)

    current_milli_time = lambda: int(round(time.time() * 1000))

    before_time = current_milli_time()

    discriminator_incorrect = 0

    epoch = 0

    if not os.path.isdir(folder+"/dcgan_output"):
        os.mkdir(folder+"/dcgan_output")
    if not os.path.isdir(folder+"/gan_models"):
        os.mkdir(folder+"/gan_models")

    print("Beginning Training.")
    print("")

    f = open(folder+"/gan_performance.txt", "a")
    
    while epoch < ABSOLUTE_EPOCHS:
        epoch_before_time = current_milli_time()
        discriminator_batch_loss = 0
        generator_batch_loss = 0
        discriminator_incorrect = 0
        if not os.path.isdir(folder+"/dcgan_output/epoch_"+str(epoch)):
            os.mkdir(folder+"/dcgan_output/epoch_"+str(epoch))
        for batch in (range(0, int(TRAIN_DATA_SIZE/BATCH_SIZE))):
            batch_before_time = current_milli_time()
            '''
            #Train discriminator
            nonseg_tensor = data[0][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].view(BATCH_SIZE, 1, B_DATA_DIMENSIONS[0], B_DATA_DIMENSIONS[1], B_DATA_DIMENSIONS[2]).to(device)
            
            #Train generator
            generator_opt.zero_grad()
            generated = generator.forward(torch.rand([BATCH_SIZE, 10]).float().to(device).half())
            generator_pred = discriminator(generated.half()).float()
            generator_label = (torch.ones(BATCH_SIZE)).float().to(device)
            generator_train_loss = torch.nn.functional.binary_cross_entropy(generator_pred.view(BATCH_SIZE), generator_label.float())
            generator_train_loss.backward()
            #for layer in generator.conv:
            #    if "ConvTranspose3d" in str(layer):
            #        print(str(layer)+" W "+str(layer.weight.grad))
                    #print(str(layer)+" B "+str(layer.bias.grad))
            generator_opt.step()
            generator_batch_loss += float(generator_train_loss)
            
            '''
            nonseg_tensor = data[0][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].view(BATCH_SIZE, 1, B_DATA_DIMENSIONS[0], B_DATA_DIMENSIONS[1], B_DATA_DIMENSIONS[2]).to(device)
            #seg_tensor = data[1][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].view(BATCH_SIZE, 1, B_DATA_DIMENSIONS[0], B_DATA_DIMENSIONS[1], B_DATA_DIMENSIONS[2]).to(device)
            #real_images = (torch.cat((nonseg_tensor, seg_tensor), dim=1).float()/256.0).to(device)
            #del nonseg_tensor
            #del seg_tensor
            fake_images = generator.forward(torch.rand([BATCH_SIZE, 100]).float().to(device).half(), epoch, batch).float()
            real_images = torch.nn.functional.interpolate(nonseg_tensor.float(), size=(DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2]))
            #del nonseg_tensor
            d_input = torch.cat((real_images, fake_images)).to(device)
            #del real_images
            #del fake_images
            discriminator_opt.zero_grad()
            decision = discriminator(d_input.half()).float().view(-1)
            #del d_input
            discriminator_label = (torch.cat((torch.rand(int(BATCH_SIZE))*LABEL_NOISE+(1-LABEL_NOISE), torch.rand(int(BATCH_SIZE))*LABEL_NOISE))).float().to(device).view(-1)
            d_incorrect = ((torch.sum(torch.abs(torch.round(decision)-torch.round(discriminator_label.float()))))/(2*BATCH_SIZE)).item()
            discriminator_incorrect += d_incorrect
            discriminator_train_loss = torch.nn.functional.binary_cross_entropy(decision.view(2*BATCH_SIZE), discriminator_label.float())
            #del decision
            #del discriminator_label
            discriminator_batch_loss += float(discriminator_train_loss)/2
            #del discriminator_train_loss

            generator_opt.zero_grad()
            generated = generator.forward(torch.rand([BATCH_SIZE, 100]).float().to(device).half())
            generator_pred = discriminator(generated.half()).float()
            #del generated
            generator_label = (torch.ones(BATCH_SIZE)).float().to(device)
            generator_train_loss = torch.nn.functional.binary_cross_entropy(generator_pred.view(BATCH_SIZE), generator_label.float())
            #del generator_pred
            generator_train_loss.backward()
            #for layer in generator.conv:
            #    if "ConvTranspose3d" in str(layer):
            #        print(str(layer)+" W "+str(layer.weight.grad))
                    #print(str(layer)+" B "+str(layer.bias.grad))
            generator_opt.step()
            generator_batch_loss += float(generator_train_loss)
            #del generator_train_loss
            

            batch_after_time = current_milli_time()
            seconds = math.floor((batch_after_time-batch_before_time)/1000)
            minutes = math.floor(seconds/60)
            seconds = seconds % 60
            #print("DBL : "+str(discriminator_batch_loss/(batch+1))+"   GBL : "+str(generator_batch_loss/(batch+1))+" Took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")
            torch.cuda.empty_cache()
            gc.collect()

        f.write(str(epoch+1)+" "+str(discriminator_batch_loss/(TRAIN_DATA_SIZE/BATCH_SIZE))+" "+str(generator_batch_loss/(TRAIN_DATA_SIZE/BATCH_SIZE))+" "+str(discriminator_incorrect/(TRAIN_DATA_SIZE/BATCH_SIZE))+"\n")
        torch.save(discriminator.state_dict(), folder+"/gan_models/dis_at_e"+str(epoch+1)+".pt")
        torch.save(generator.state_dict(), folder+"/gan_models/gen_at_e"+str(epoch+1)+".pt")
        generated = generator.forward(torch.rand(NUM_GENERATED_PER_EPOCH, 100).float().to(device).half()).float()
        for image in range(0, NUM_GENERATED_PER_EPOCH):
            for dim in range(0, generated.size()[2]):
                save_image(generated[image, 0, dim, :, :], folder+"/dcgan_output/epoch_"+str(epoch)+"/nonseg_"+str(dim+1)+".png")
                #save_image(generated[image, 1, dim, :, :], folder+"/dcgan_output/epoch_"+str(epoch)+"/seg_"+str(dim+1)+".png")
        del generated

        epoch_after_time = current_milli_time()
        seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60

        print("")
        print("Discriminator Epoch "+str(epoch+1)+" Loss : "+str(discriminator_batch_loss/(TRAIN_DATA_SIZE/BATCH_SIZE)))
        print("Generator Epoch "+str(epoch+1)+" Loss : "+str(generator_batch_loss/(TRAIN_DATA_SIZE/BATCH_SIZE)))
        print("Portion Discriminator Incorrect : "+str(discriminator_incorrect/(TRAIN_DATA_SIZE/BATCH_SIZE)))
        print("Took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")
        print("")

        epoch += 1

    f.write("\n")
    f.close()

    after_time = current_milli_time()

    seconds = math.floor((after_time-before_time)/1000)
    minutes = math.floor(seconds/60)
    seconds = seconds % 60

    print(str(epoch)+" epochs took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")
    return generator, discriminator

if __name__ == "__main__":
    print("Start!")
    before_time = current_milli_time()

    files = os.listdir(".")
    m = [int(f[8:]) for f in files if len(f) > 8 and f[0:8] == "gantrial"]
    if len(m) > 0:
        folder = "gantrial"+str(max(m)+1)
    else:
        folder = "gantrial1"
    os.mkdir(folder)

    print("Created session folder "+folder)

    print("Loading data...")
    data = torch.load("TRIMMED_DATA.pt")

    after_time = current_milli_time()
    seconds = math.floor((after_time-before_time)/1000)
    minutes = math.floor(seconds/60)
    seconds = seconds % 60
    print("Data loading took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")

    generator, discriminator = train_model(data)
