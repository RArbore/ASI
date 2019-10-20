import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def display_images(image_file, factor_of_ten):
    multiplier = 10**(factor_of_ten-1)
    image_file = open(image_file, "rb")
    byte_array = []
    byte = image_file.read(1)
    '''
    count = 0
    while count < 16:
        image_file.read(1)
        count += 1
    '''
    big_count = 0
    fig=plt.figure(figsize=(8, 2*multiplier))
    while big_count < 10*multiplier:
        count = 0
        byte_array = []
        while byte and count < 784:
            byte_array.append(int.from_bytes(byte, byteorder="big"))
            byte = image_file.read(1)
            count += 1
        tensor = torch.from_numpy(np.asarray(byte_array))
        tensor = tensor.reshape(28, 28).float()
        tensor = tensor/256
        #tensor += torch.randn(tensor.size())/5
        tensor = (tensor*256).int()
        tensor = torch.min(tensor, (torch.ones(tensor.size())*255).int())
        tensor = torch.max(tensor, (torch.zeros(tensor.size())).int())
        image = tensor.clone().cpu()
        image = image.view(*tensor.size())
        image = transforms.ToPILImage()(image)
        fig.add_subplot(2*multiplier, 5, big_count+1)
        plt.imshow(image)
        big_count += 1
        if big_count%100 == 0:
            print("100 images loaded.")
    print("Drawing...")
    plt.show()