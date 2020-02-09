import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import math
import time

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

train_data_size = 335

batch_size = 5

image_size = 64

nc = 1

nz = 100

ngf = 64

ndf = 64

num_epochs = 2400

lr = 0.0002

beta1 = 0.5

ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def current_milli_time():
    return int(round(time.time() * 1000))

before_time = current_milli_time()

files = os.listdir(".")
m = [int(f[8:]) for f in files if len(f) > 8 and f[0:8] == "gantrial"]
if len(m) > 0:
    folder = "gantrial" + str(max(m) + 1)
else:
    folder = "gantrial1"
os.mkdir(folder)

print("Created session folder " + folder)

print("Loading data...")
#data = nn.functional.interpolate(torch.load("TRIMMED_DATA.pt").float(), size=(image_size, image_size, image_size))
data = torch.load("TRIMMED64.pt")

after_time = current_milli_time()
seconds = math.floor((after_time - before_time) / 1000)
minutes = math.floor(seconds / 60)
seconds = seconds % 60
print("Data loading took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose3d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

netD = Discriminator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, 1, device=device)

real_label = 0.8
fake_label = 0.2

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []

def save_image(tensor, filename):
    ndarr = tensor.mul(256).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)

if not os.path.isdir(folder + "/dcgan_output"):
    os.mkdir(folder + "/dcgan_output")
if not os.path.isdir(folder + "/gan_models"):
    os.mkdir(folder + "/gan_models")
f = open(folder + "/gan_performance.txt", "a")

D_G_z2_epoch = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    epoch_before_time = current_milli_time()
    if not os.path.isdir(folder + "/dcgan_output/epoch_" + str(epoch)):
        os.mkdir(folder + "/dcgan_output/epoch_" + str(epoch))
    for batch in range(int(train_data_size/batch_size)):
        nonseg_tensor = data[0][batch * batch_size:(batch + 1) * batch_size].view(batch_size, 1,  image_size, image_size, image_size).to(device)
        
        label = torch.full((batch_size,), real_label, device=device)
        noise = torch.randn(batch_size, nz, 1, 1, 1, device=device)
        fake = netG(noise)
        if epoch < 2 or D_G_z2_epoch > real_label*(3/4):
            netD.zero_grad()
            real_cpu = nonseg_tensor.to(device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
    
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if batch == 66:
            D_G_z2_epoch = D_G_z2
            epoch_after_time = current_milli_time()
            seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
            minutes = math.floor(seconds / 60)
            seconds = seconds % 60
            print(('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'% (epoch, num_epochs, batch, train_data_size/batch_size, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))+"   Time: "+str(minutes)+" minute(s) and "+str(seconds)+" second(s)")
            f.write(str(epoch + 1) + " " + str(errD.item()) + " " + str(errG.item()) + " " + str(D_x) + " " + str(D_G_z1) + " " + str(D_G_z2) + "\n")
            torch.save(netD.state_dict(), folder + "/gan_models/dis_at_e" + str(epoch + 1) + ".pt")
            torch.save(netG.state_dict(), folder + "/gan_models/gen_at_e" + str(epoch + 1) + ".pt")
            for image in range(0, batch_size):
                for dim in range(0, fake.size()[2]):
                    save_image(fake[image, 0, dim, :, :], folder + "/dcgan_output/epoch_" + str(epoch) + "/nonseg_image" + str(image + 1) + "_num" + str(dim + 1) + ".png")

        G_losses.append(errG.item())
        D_losses.append(errD.item())
