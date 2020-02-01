import torch
from torchvision import transforms
import time
import math
import os
import gc

#Define constants.

DATA_SIZE = 149*173*192

DATA_DIMENSIONS = [149, 173, 192]

TRAIN_DATA_SIZE = 335

BATCH_SIZE = 5

MIN_EPOCHS = 50

ABSOLUTE_EPOCHS = 1000

NUM_GENERATED_PER_EPOCH = 10

MAX_BLOCKS = 5

LABEL_NOISE = 0.25

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

folder = ""


#Define the networks.

class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv3d(100, 90, (2, 2, 2), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(90, 70, (3, 3, 3), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv3d(70, 50, (3, 3, 3), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv3d(50, 30, (3, 3, 3), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
        )
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv3d(30, 10, (3, 3, 3), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
        )
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv3d(10, 2, (3, 3, 3), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
        )

        self.blocks = [self.block1, self.block2, self.block3, self.block4, self.block5]
        self.interpolate_sizes = [(6, 6, 6), (18, 18, 18), (54, 54, 54), (DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2]), (DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2])]

        self.acti = torch.nn.Sequential(
            torch.nn.Tanh(),
        )

        self.rgb1 = torch.nn.Sequential(
            torch.nn.Conv1d(70, 2, (1, 1, 1)),
        )
        self.rgb2 = torch.nn.Sequential(
            torch.nn.Conv1d(50, 2, (1, 1, 1)),
        )
        self.rgb3 = torch.nn.Sequential(
            torch.nn.Conv1d(30, 2, (1, 1, 1)),
        )
        self.rgb4 = torch.nn.Sequential(
            torch.nn.Conv1d(10, 2, (1, 1, 1)),
        )

        self.to_rgb_layers = [self.rgb1, self.rgb2, self.rgb3, self.rgb4]
        self.rgb_index = [70, 50, 30, 10, 2]

    def to_rgb(self, input):
        index = self.rgb_index.index(input.size()[1])
        if index == 4:
            return input
        return self.to_rgb_layers[index](input)

    def forward(self, input, BLOCKS, ALPHA, epoch):
        out = input.view(-1, 100, 1, 1, 1)
        
        for i in range(0, BLOCKS):
            if i == BLOCKS-1 and BLOCKS > 1:
                before = self.to_rgb(torch.nn.functional.interpolate(out, size=self.interpolate_sizes[i], mode="trilinear", align_corners=False))
                out = self.to_rgb(torch.nn.functional.interpolate(self.blocks[i](out), size=self.interpolate_sizes[i], mode="trilinear", align_corners=False))
                out = before*(1-ALPHA)+out*(ALPHA)
            elif i == BLOCKS-1:
                out = self.to_rgb(torch.nn.functional.interpolate(self.blocks[i](out), size=self.interpolate_sizes[i], mode="trilinear", align_corners=False))
            else:
                out = torch.nn.functional.interpolate(self.blocks[i](out), size=self.interpolate_sizes[i], mode="trilinear", align_corners=False)

        out = (self.acti(out)+1)/2
        return out * torch.tensor(0.998) + torch.tensor(0.001)

class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 2, (3, 3, 3), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 2, (3, 3, 3), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 2, (3, 3, 3), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
        )
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 2, (3, 3, 3), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
        )
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 2, (3, 3, 3), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(2, 2, (2, 2, 2)),
        )

        self.blocks = [self.block1, self.block2, self.block3, self.block4, self.block5]
        self.interpolate_sizes = [(54, 54, 54), (18, 18, 18), (6, 6, 6), (2, 2, 2), (1, 1, 1)]

        self.acti = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, input, BLOCKS, ALPHA):
        out = input

        for i in range(len(self.blocks)-BLOCKS, len(self.blocks)):
            out = torch.nn.functional.interpolate(self.blocks[i](out), size=self.interpolate_sizes[i], mode="trilinear", align_corners=False)

        out = self.acti(out.view(-1, 2))
        return out.view(-1, 2)*torch.tensor(0.998)+torch.tensor(0.001)


#Define a function for saving images.

def save_image(tensor, filename):
    ndarr = tensor.mul(256).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)


#Define pixel-wise binary cross entropy.

def pixel_BCE(output_tensor, label_tensor):
    loss_tensor = label_tensor*torch.log(output_tensor)+(torch.ones(label_tensor.size()).float().to(device)-label_tensor)*torch.log(torch.ones(output_tensor.size()).float().to(device)-output_tensor)
    return torch.mean(loss_tensor)*torch.tensor(-1).to(device)


#Declare and train the network.

def train_model(data):
    ALPHA = 0
    BLOCKS = 1

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

    while epoch < ABSOLUTE_EPOCHS:# and (discriminator_incorrect < 0.4 or epoch < MIN_EPOCHS):
        epoch_before_time = current_milli_time()
        discriminator_batch_loss = 0
        generator_batch_loss = 0
        discriminator_incorrect = 0
        if not os.path.isdir(folder+"/dcgan_output/epoch_"+str(epoch)):
            os.mkdir(folder+"/dcgan_output/epoch_"+str(epoch))
        for batch in (range(0, int(TRAIN_DATA_SIZE/BATCH_SIZE))):
            batch_before_time = current_milli_time()
            nonseg_tensor = data[0][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].view(BATCH_SIZE, 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2]).to(device)
            seg_tensor = data[1][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].view(BATCH_SIZE, 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2]).to(device)
            real_images = (torch.stack((nonseg_tensor, seg_tensor), dim=1).float()/256.0).to(device)
            del nonseg_tensor
            del seg_tensor
            fake_images = generator.forward(torch.rand([BATCH_SIZE, 100]).float().to(device).half(), BLOCKS, ALPHA, epoch).float()
            real_images = torch.nn.functional.interpolate(real_images.view(BATCH_SIZE, 2, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2]), size=fake_images.size()[2:5])
            d_input = torch.cat((real_images, fake_images)).to(device)
            del real_images
            del fake_images
            discriminator_opt.zero_grad()
            decision = discriminator(d_input.half(), BLOCKS, ALPHA).float()
            del d_input
            discriminator_label = torch.stack((torch.cat((torch.rand(int(BATCH_SIZE))*LABEL_NOISE+(1-LABEL_NOISE), torch.rand(int(BATCH_SIZE))*LABEL_NOISE)), torch.cat((torch.rand(int(BATCH_SIZE))*LABEL_NOISE, torch.rand(int(BATCH_SIZE))*LABEL_NOISE+(1-LABEL_NOISE)))), dim=1).float().to(device)
            discriminator_incorrect += ((torch.sum(torch.abs(torch.round(decision)-discriminator_label.float())))/(4*BATCH_SIZE)).item()
            discriminator_train_loss = pixel_BCE(decision, discriminator_label.float())
            del decision
            del discriminator_label
            discriminator_train_loss.backward()
            discriminator_opt.step()
            discriminator_batch_loss += float(discriminator_train_loss)/2
            del discriminator_train_loss

            generator_opt.zero_grad()
            generated = generator.forward(torch.rand([BATCH_SIZE, 100]).float().to(device).half(), BLOCKS, ALPHA, epoch)
            generator_pred = discriminator(generated.half(), BLOCKS, ALPHA).float()
            del generated
            generator_label = torch.stack((torch.ones(BATCH_SIZE), torch.zeros(BATCH_SIZE)), dim=1).float().to(device)
            generator_train_loss = pixel_BCE(generator_pred, generator_label.float())
            del generator_pred
            generator_train_loss.backward()
            generator_opt.step()
            generator_batch_loss += float(generator_train_loss)
            del generator_train_loss

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
        generated = generator.forward(torch.rand(NUM_GENERATED_PER_EPOCH, 100).float().to(device).half(), BLOCKS, ALPHA, -1).float()
        for image in range(0, NUM_GENERATED_PER_EPOCH):
            for dim in range(0, generated.size()[2]):
                save_image(generated[image, 0, dim, :, :], folder+"/dcgan_output/epoch_"+str(epoch)+"/nonseg_"+str(dim+1)+".png")
                save_image(generated[image, 1, dim, :, :], folder+"/dcgan_output/epoch_"+str(epoch)+"/seg_"+str(dim+1)+".png")
        del generated

        if BLOCKS <= MAX_BLOCKS:
            #ALPHA += 0.01
            if ALPHA > 1:
                if BLOCKS < MAX_BLOCKS:
                    ALPHA = 0
                    BLOCKS += 1
                else:
                    ALPHA = 1

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
    current_milli_time = lambda: int(round(time.time() * 1000))
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
