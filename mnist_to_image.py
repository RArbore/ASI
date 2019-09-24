import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

#image_file = open("MNIST_TRAIN_IMAGES", "rb")
image_file = open("AE_GENERATED_IMAGES", "rb")
byte_array = []
byte = image_file.read(1)
'''
count = 0
while count < 16:
    image_file.read(1)
    count += 1
'''
big_count = 0
fig=plt.figure(figsize=(7, 3))
while big_count < 10:
    count = 0
    byte_array = []
    while byte and count < 784:
        byte_array.append(int.from_bytes(byte, byteorder="big"))
        byte = image_file.read(1)
        count += 1
    tensor = torch.from_numpy(np.asarray(byte_array))
    tensor = tensor.reshape(28, 28).int()
    image = tensor.clone().cpu()
    image = image.view(*tensor.size())
    image = transforms.ToPILImage()(image)
    fig.add_subplot(2, 5, big_count+1)
    plt.imshow(image)
    big_count += 1
plt.show()